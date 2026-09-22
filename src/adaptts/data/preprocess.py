"""Offline preprocessing: turn raw audio and text into cached training tensors.

Stages, each resumable and each writing a memmap or JSON to the cache:

  A0  manifest      - scan the dataset, normalize text, filter by duration
  A1  align         - CTC forced alignment  -> word time spans
  A2  span_embed    - SSL embeddings of each word span
  A3  discover      - cluster spans -> pronunciation codes + per-token labels
  A4  teacher       - frozen MARBERTv2 contextual states + teacher code head
  A5  codec         - Mimi encode -> RVQ codes

Everything the training loop needs is a memmap read after this runs, so the GPU
is never waiting on audio decoding, BERT or the codec.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..text.egyptian import normalize_egyptian
from ..text.normalize import normalize_text, tokenize_words
from ..text.vocab import CharVocab
from ..utils.config import Config
from ..utils.logging_utils import log_table, progress
from .ctc_aligner import (
    WordAlignment,
    ctc_forced_align,
    enforce_monotonic,
    spans_from_path,
    word_spans_from_char_spans,
)
from .discovery import PronunciationLexicon, WordCodes, discover_word_codes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A0: manifest
# ---------------------------------------------------------------------------


@dataclass
class Utterance:
    uid: str
    audio_path: str
    text: str
    duration: float
    split: str
    speaker: str = "default"

    def to_json(self) -> dict:
        return {
            "uid": self.uid,
            "audio_path": self.audio_path,
            "text": self.text,
            "duration": self.duration,
            "split": self.split,
            "speaker": self.speaker,
        }


def _read_metadata_csv(path: Path) -> List[dict]:
    import csv

    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_parquet_dataset(
    root: Path, cfg: Config, text_column: Optional[str] = None
) -> List[dict]:
    """Unpack a HuggingFace parquet dataset into wav files plus metadata rows.

    Datasets published as parquet embed the audio bytes in the table. Decoding
    those bytes on every epoch would dominate training time, so we extract once
    to wav and let every later stage memory-map the results.

    The extraction is skipped when the clips already exist, so rerunning the
    stage is cheap.
    """
    import soundfile as sf

    files = sorted(root.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no .parquet files under {root}")

    clips_dir = root / "clips" / "parquet"
    clips_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    index = 0

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "reading a parquet dataset needs pyarrow: pip install pyarrow"
        ) from exc

    logger.info("parquet: %d shard(s) under %s", len(files), root)

    for shard in progress(files, desc="extracting parquet", unit="shard"):
        table = pq.read_table(shard)
        names = set(table.column_names)

        audio_col = next((c for c in ("audio", "wav", "speech") if c in names), None)
        if audio_col is None:
            raise ValueError(
                f"{shard.name} has no audio column; found {sorted(names)}"
            )
        # Prefer an explicitly requested column, else the most common names.
        candidates = ([text_column] if text_column else []) + [
            "text", "transcription", "transcript", "sentence", "cohere_text",
        ]
        txt_col = next((c for c in candidates if c and c in names), None)
        if txt_col is None:
            raise ValueError(
                f"{shard.name} has no text column; found {sorted(names)}"
            )

        data = table.to_pylist()
        for rec in progress(data, desc=f"  {shard.name[:18]}", unit="clip", leave=False):
            audio = rec.get(audio_col)
            text = (rec.get(txt_col) or "").strip()
            if not text or audio is None:
                continue

            uid = f"pq_{index:06d}"
            out_wav = clips_dir / f"{uid}.wav"
            index += 1

            if out_wav.exists():
                try:
                    info = sf.info(str(out_wav))
                    rows.append({
                        "file_name": str(out_wav.relative_to(root)).replace("\\", "/"),
                        "id": uid, "text": text,
                        "duration": info.frames / info.samplerate,
                        "split": "", "channel": "parquet",
                    })
                    continue
                except Exception:  # noqa: BLE001
                    pass  # unreadable: fall through and rewrite it

            try:
                if isinstance(audio, dict) and audio.get("bytes"):
                    import io as _io

                    wav, sr = sf.read(_io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
                elif isinstance(audio, dict) and audio.get("path"):
                    wav, sr = sf.read(audio["path"], dtype="float32", always_2d=True)
                elif isinstance(audio, dict) and audio.get("array") is not None:
                    wav = np.asarray(audio["array"], dtype=np.float32).reshape(-1, 1)
                    sr = int(audio.get("sampling_rate") or cfg.audio.sample_rate)
                else:
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not decode %s: %s", uid, exc)
                continue

            mono = wav.mean(axis=1)
            if sr != cfg.audio.sample_rate:
                import torch as _torch
                import torchaudio as _ta

                mono = _ta.functional.resample(
                    _torch.from_numpy(mono), sr, cfg.audio.sample_rate
                ).numpy()
                sr = cfg.audio.sample_rate

            sf.write(out_wav, mono, sr)
            rows.append({
                "file_name": str(out_wav.relative_to(root)).replace("\\", "/"),
                "id": uid, "text": text, "duration": len(mono) / sr,
                "split": "", "channel": "parquet",
            })

    logger.info("parquet: extracted %d clips to %s", len(rows), clips_dir)
    return rows


def _assign_splits(
    rows: List[dict], dev_frac: float = 0.04, test_frac: float = 0.04, seed: int = 1234
) -> None:
    """Assign train/dev/test in place for datasets that ship only one split.

    Deterministic, so reruns keep the same partition and a checkpoint evaluated
    on "dev" always means the same utterances.
    """
    unsplit = [r for r in rows if not r.get("split")]
    if not unsplit:
        return
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unsplit))
    n_dev = max(1, int(len(unsplit) * dev_frac))
    n_test = max(1, int(len(unsplit) * test_frac))
    for rank, idx in enumerate(order):
        if rank < n_dev:
            unsplit[idx]["split"] = "dev"
        elif rank < n_dev + n_test:
            unsplit[idx]["split"] = "test"
        else:
            unsplit[idx]["split"] = "train"
    logger.info(
        "split: %d train / %d dev / %d test (auto-assigned, seed=%d)",
        len(unsplit) - n_dev - n_test, n_dev, n_test, seed,
    )


def build_manifest(cfg: Config, force: bool = False) -> List[Utterance]:
    """Scan the dataset directory and produce a normalized manifest.

    Supports the Masri-100h layout (``metadata.csv`` plus ``metadata/*.jsonl``)
    and any dataset exposing the same columns. Text is normalized once here so
    that every later stage sees identical strings.
    """
    out_path = Path(cfg.paths.manifest_path)
    if out_path.exists() and not force:
        rows = _read_jsonl(out_path)
        logger.info("manifest: reusing %d utterances from %s", len(rows), out_path)
        return [Utterance(**r) for r in rows]

    root = Path(cfg.paths.dataset_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"dataset directory not found: {root}. Download it first "
            f"(see notebooks/01_prepare_data.ipynb)."
        )

    rows: List[dict] = []
    meta_dir = root / "metadata"
    parquet_files = list(root.glob("**/*.parquet"))

    if parquet_files and not (meta_dir.exists() and any(meta_dir.glob("*.jsonl"))):
        # A HuggingFace parquet dataset (audio embedded in the table).
        rows = _extract_parquet_dataset(root, cfg, cfg.text.parquet_text_column or None)
    elif meta_dir.exists() and any(meta_dir.glob("*.jsonl")):
        for split in ("train", "dev", "test"):
            p = meta_dir / f"{split}.jsonl"
            if p.exists():
                for r in _read_jsonl(p):
                    r.setdefault("split", split)
                    rows.append(r)
        if not rows:
            rows = _read_jsonl(meta_dir / "all.jsonl")
    elif (root / "metadata.csv").exists():
        rows = _read_metadata_csv(root / "metadata.csv")
    else:
        raise FileNotFoundError(
            f"no metadata found under {root}; expected metadata/*.jsonl or metadata.csv"
        )

    _assign_splits(rows)

    utts: List[Utterance] = []
    skipped = defaultdict(int)
    durations_seen: List[float] = []
    for r in progress(rows, desc="building manifest", unit="utt"):
        rel = r.get("file_name") or r.get("audio_path") or r.get("path")
        if not rel:
            skipped["no_path"] += 1
            continue
        apath = (root / rel) if not os.path.isabs(rel) else Path(rel)
        if not apath.exists():
            skipped["missing_audio"] += 1
            continue

        raw = r.get("text") or r.get("transcript") or ""
        if cfg.text.use_egyptian_normalizer:
            # Expand numbers, dates, URLs and abbreviations into spoken words
            # first, so the acoustic model never sees a digit or a Latin token.
            raw = normalize_egyptian(
                raw,
                expand_numbers=True,
                expand_latin=True,
                keep_punctuation=False,
                strip_marks=cfg.text.strip_diacritics,
            )
        text = normalize_text(
            raw,
            strip_diacritics_flag=cfg.text.strip_diacritics,
            lowercase_latin=cfg.text.lowercase_latin,
            normalize_alef=cfg.text.normalize_alef,
            normalize_digits=cfg.text.normalize_digits,
        )
        if not text:
            skipped["empty_text"] += 1
            continue
        if len(text) > cfg.text.max_chars:
            skipped["too_long_text"] += 1
            continue

        try:
            dur = float(r.get("duration") or 0.0)
        except (TypeError, ValueError):
            dur = 0.0
        if dur <= 0:
            import soundfile as sf

            info = sf.info(str(apath))
            dur = info.frames / info.samplerate
        durations_seen.append(dur)
        if not (cfg.audio.min_duration <= dur <= cfg.audio.max_duration):
            skipped["duration"] += 1
            continue

        words, _ = tokenize_words(text)
        if not words or len(words) > cfg.text.max_words:
            skipped["word_count"] += 1
            continue

        uid = str(r.get("id") or Path(rel).stem)
        utts.append(
            Utterance(
                uid=uid,
                audio_path=str(apath),
                text=text,
                duration=dur,
                split=str(r.get("split") or "train"),
                speaker=str(r.get("channel") or r.get("speaker") or "default"),
            )
        )

    if not utts:
        lines = [
            "every utterance was filtered out.",
            f"  reasons: {dict(skipped)}",
        ]
        if durations_seen:
            arr = np.array(durations_seen)
            lines += [
                f"  corpus durations: min {arr.min():.1f}s  "
                f"median {float(np.median(arr)):.1f}s  max {arr.max():.1f}s",
                f"  your window:      {cfg.audio.min_duration:.1f}s to "
                f"{cfg.audio.max_duration:.1f}s",
                "  widen audio.min_duration / audio.max_duration to cover the corpus.",
            ]
        raise RuntimeError(chr(10).join(lines))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for u in utts:
            f.write(json.dumps(u.to_json(), ensure_ascii=False) + "\n")

    counts: Dict[str, int] = defaultdict(int)
    hours: Dict[str, float] = defaultdict(float)
    for u in utts:
        counts[u.split] += 1
        hours[u.split] += u.duration / 3600.0

    log_table(
        logger,
        ["split", "clips", "hours"],
        [(k, counts[k], f"{hours[k]:.2f}") for k in sorted(counts)]
        + [("TOTAL", len(utts), f"{sum(hours.values()):.2f}")],
    )
    if skipped:
        logger.info("skipped during filtering:")
        log_table(logger, ["reason", "count"], sorted(skipped.items()))

    # Losing a large share of a corpus is almost always a misconfigured filter,
    # not a property of the data. Say so loudly: a duration cap below the
    # corpus median once discarded 71% of the Masri set while the run still
    # reported success.
    total_seen = len(utts) + sum(skipped.values())
    kept_frac = len(utts) / max(total_seen, 1)
    if kept_frac < 0.9 and total_seen:
        worst = max(skipped.items(), key=lambda kv: kv[1])
        logger.warning("")
        logger.warning("=" * 68)
        logger.warning(
            "KEPT ONLY %.0f%% OF THE CORPUS (%d of %d clips)",
            100 * kept_frac, len(utts), total_seen,
        )
        logger.warning("largest cause: %s (%d clips)", worst[0], worst[1])
        if worst[0] == "duration":
            logger.warning("")
            logger.warning(
                "Your audio.min_duration / audio.max_duration window is cutting "
                "real data."
            )
            logger.warning(
                "Current window: %.1f s to %.1f s", cfg.audio.min_duration,
                cfg.audio.max_duration,
            )
            if durations_seen:
                arr = np.array(durations_seen)
                logger.warning(
                    "Corpus durations: min %.1f  median %.1f  max %.1f",
                    arr.min(), float(np.median(arr)), arr.max(),
                )
                logger.warning(
                    "Set audio.max_duration above %.1f to keep everything.",
                    arr.max(),
                )
        logger.warning("=" * 68)
        logger.warning("")

    return utts


def build_char_vocab(cfg: Config, utts: Sequence[Utterance], force: bool = False) -> CharVocab:
    path = Path(cfg.paths.charvocab_path)
    if path.exists() and not force:
        return CharVocab.load(path)
    vocab = CharVocab.build((u.text for u in utts if u.split == "train"), min_freq=2)
    vocab.save(path)
    logger.info("char vocab: %d symbols -> %s", len(vocab), path)
    return vocab


# ---------------------------------------------------------------------------
# A1: CTC forced alignment
# ---------------------------------------------------------------------------


class CTCAlignerRunner:
    """Word-level forced alignment with a CTC acoustic model.

    The default aligner is ``MahmoudAshraf/mms-300m-1130-forced-aligner``, a
    massively multilingual model whose CTC vocabulary is **romanized**: 31
    Latin tokens, no Arabic script. So the pipeline is

        Arabic text -> romanize (keeping a character index map)
                    -> CTC target ids
                    -> Viterbi alignment against the acoustic posteriors
                    -> Latin frame spans
                    -> Arabic character spans (via the index map)
                    -> word spans

    The romanization is a script mapping used only to find frame boundaries. It
    makes no pronunciation decisions; those stay with the discovered codes.

    The model is also loaded with safetensors, which matters: transformers 5.x
    refuses to load ``.bin`` checkpoints unless torch is at least 2.6
    (CVE-2025-32434), and most Arabic wav2vec2 checkpoints ship only ``.bin``.
    """

    def __init__(self, cfg: Config, device: torch.device) -> None:
        from transformers import AutoModelForCTC, AutoProcessor

        self.cfg = cfg
        self.device = device
        self.processor = AutoProcessor.from_pretrained(cfg.align.model_id)
        self.model = (
            AutoModelForCTC.from_pretrained(cfg.align.model_id).to(device).eval()
        )
        tok = self.processor.tokenizer
        self.vocab: Dict[str, int] = tok.get_vocab()
        self.sample_rate = self.processor.feature_extractor.sampling_rate

        # The CTC blank is not always the tokenizer's pad token. This model has
        # an explicit <blank> at index 0 while pad_token_id is 1, and using the
        # wrong one silently produces garbage alignments.
        self.blank_id = self._find_blank(tok)

        # Detect whether the vocabulary is romanized or native Arabic script, so
        # a different aligner can be swapped in through the config.
        arabic_letters = sum(
            1 for c in "\u0627\u0628\u062a\u062c\u062f\u0631\u0633\u0645\u0646"
            if c in self.vocab
        )
        self.romanized = arabic_letters < 5
        logger.info(
            "aligner: %s | vocab %d | blank id %d | %s targets",
            cfg.align.model_id, len(self.vocab), self.blank_id,
            "romanized" if self.romanized else "arabic-script",
        )

    @staticmethod
    def _find_blank(tok) -> int:
        vocab = tok.get_vocab()
        for name in ("<blank>", "<pad>", "<s>"):
            if name in vocab:
                return vocab[name]
        if tok.pad_token_id is not None:
            return int(tok.pad_token_id)
        return 0

    def _targets(self, text: str) -> Tuple[List[int], List[int]]:
        """CTC target ids plus, for each, the index of its source character."""
        if self.romanized:
            from ..text.romanize import build_ctc_targets

            return build_ctc_targets(text, self.vocab, self.vocab.get("<unk>", 3))

        ids: List[int] = []
        src: List[int] = []
        for i, ch in enumerate(text):
            if ch == " ":
                for cand in ("|", "<space>", " "):
                    if cand in self.vocab:
                        ids.append(self.vocab[cand])
                        src.append(i)
                        break
                continue
            tid = self.vocab.get(ch) or self.vocab.get(ch.upper())
            if tid is None:
                continue
            ids.append(tid)
            src.append(i)
        return ids, src

    @torch.no_grad()
    def align(self, waveform: torch.Tensor, text: str) -> Tuple[List[WordAlignment], float]:
        """Align one utterance. ``waveform`` is 1-D at ``self.sample_rate``."""
        words, spans = tokenize_words(text)
        if not words:
            return [], 0.0

        ids, src_index = self._targets(text)
        if len(ids) < 2:
            return [], 0.0

        # Which word each source character belongs to.
        char_to_word: Dict[int, int] = {}
        for wi, (a, b) in enumerate(spans):
            for ci in range(a, b):
                char_to_word[ci] = wi
        word_of_target = [char_to_word.get(i, -1) for i in src_index]

        logits = self.model(waveform[None].to(self.device)).logits[0]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        fps = logits.shape[0] / (waveform.shape[-1] / self.sample_rate)

        try:
            path, scores = ctc_forced_align(log_probs, ids, blank=self.blank_id)
        except (ValueError, RuntimeError):
            # Too few frames for the target length, or a degenerate lattice.
            return [], 0.0

        target_spans = spans_from_path(path, scores, len(ids))
        aligns = word_spans_from_char_spans(target_spans, word_of_target, words)
        return enforce_monotonic(aligns), fps


# ---------------------------------------------------------------------------
# A2 + A3: span embeddings and pronunciation-code discovery
# ---------------------------------------------------------------------------


class SpanEmbedder:
    """Mean-pooled self-supervised features over a word's time span."""

    def __init__(self, cfg: Config, device: torch.device) -> None:
        from transformers import AutoModel, AutoFeatureExtractor

        self.cfg = cfg
        self.device = device
        self.fe = AutoFeatureExtractor.from_pretrained(cfg.spanemb.model_id)
        self.model = (
            AutoModel.from_pretrained(cfg.spanemb.model_id).to(device).eval()
        )
        self.layers = tuple(cfg.spanemb.layers)
        self.sample_rate = self.fe.sampling_rate

    @torch.no_grad()
    def embed(
        self, waveform: torch.Tensor, spans_sec: Sequence[Tuple[float, float]]
    ) -> np.ndarray:
        """Return ``[n_spans, dim]`` embeddings for the given time spans."""
        if not spans_sec:
            return np.zeros((0, self.cfg.spanemb.dim), dtype=np.float32)

        out = self.model(waveform[None].to(self.device), output_hidden_states=True)
        hs = out.hidden_states
        feats = torch.stack([hs[l] for l in self.layers], dim=0).mean(0)[0]  # [T, D]
        n_frames = feats.shape[0]
        secs = waveform.shape[-1] / self.sample_rate
        fps = n_frames / max(secs, 1e-6)

        # Utterance-level mean is removed so the embedding describes *this
        # word's* realization rather than the speaker's session-level timbre.
        utt_mean = feats.mean(dim=0, keepdim=True)

        pad = self.cfg.spanemb.context_pad
        vecs = []
        for t0, t1 in spans_sec:
            a = int(max(0, (t0 - pad) * fps))
            b = int(min(n_frames, np.ceil((t1 + pad) * fps)))
            if b <= a:
                b = min(n_frames, a + 1)
            if b <= a:
                vecs.append(torch.zeros(feats.shape[1], device=feats.device))
            else:
                vecs.append((feats[a:b] - utt_mean).mean(dim=0))
        return torch.stack(vecs).float().cpu().numpy()


@dataclass
class OccurrenceRecord:
    """One observed use of one word type."""

    uid: str
    word_index: int
    word: str
    start_sec: float
    end_sec: float
    score: float
    row: int  # index into the span-embedding memmap


def _discovery_params(cfg: Config) -> dict:
    """Flatten the discovery settings into a picklable dict for the workers."""
    d = cfg.discovery
    return {
        "max_codes": d.max_codes_per_word,
        "n_bootstrap": d.n_bootstrap,
        "bootstrap_frac": d.bootstrap_frac,
        "stability_threshold": d.stability_threshold,
        "silhouette_threshold": d.silhouette_threshold,
        "min_separation": d.min_separation,
        "min_cluster_frac": d.min_cluster_frac,
        "max_duration_confound": d.max_duration_confound,
        "pca_dim": d.pca_dim,
        "seed": d.random_seed,
    }


def _discover_one(args: tuple) -> Tuple[str, Optional[WordCodes]]:
    """Discover the codes for one word type. Runs inside a worker process.

    Defined at module level so it survives pickling on spawn-based platforms.
    Each call receives its own feature block, so workers share no state.
    """
    word, feats, durs, params = args
    wc = discover_word_codes(
        word, feats,
        durations=durs,
        max_codes=params["max_codes"],
        n_bootstrap=params["n_bootstrap"],
        bootstrap_frac=params["bootstrap_frac"],
        stability_threshold=params["stability_threshold"],
        silhouette_threshold=params["silhouette_threshold"],
        min_separation=params["min_separation"],
        min_cluster_frac=params["min_cluster_frac"],
        max_duration_confound=params["max_duration_confound"],
        pca_dim=params["pca_dim"],
        seed=params["seed"],
    )
    return word, (wc if wc.n_codes > 1 else None)


def _build_job(
    cfg: Config,
    word: str,
    idxs: Sequence[int],
    occurrences: Sequence[OccurrenceRecord],
    embeddings: np.ndarray,
    rng: np.random.Generator,
    params: dict,
) -> tuple:
    """Gather one word's features and durations into a self-contained job."""
    d = cfg.discovery
    sel = list(idxs)
    if len(sel) > d.max_occurrences_per_word:
        sel = list(rng.choice(sel, size=d.max_occurrences_per_word, replace=False))
    feats = np.asarray(
        embeddings[[occurrences[i].row for i in sel]], dtype=np.float32
    )
    durs = np.array(
        [occurrences[i].end_sec - occurrences[i].start_sec for i in sel],
        dtype=np.float64,
    )
    return (word, feats, durs, params)


def discover_pronunciation_codes(
    cfg: Config,
    occurrences: Sequence[OccurrenceRecord],
    embeddings: np.ndarray,
    force: bool = False,
) -> Tuple[PronunciationLexicon, Dict[Tuple[str, int], int]]:
    """Run discovery for every sufficiently frequent word type.

    Returns the lexicon and a mapping ``(uid, word_index) -> code``, which is
    the supervision the context encoder trains on.

    Word types are independent, so the search runs across worker processes.
    This stage uses no GPU, and on a rented pod the CPU cores would otherwise
    sit idle behind a single-threaded loop.
    """
    path = Path(cfg.paths.lexicon_path)
    d = cfg.discovery

    by_word: Dict[str, List[int]] = defaultdict(list)
    for i, occ in enumerate(occurrences):
        by_word[occ.word].append(i)

    if path.exists() and not force:
        lex = PronunciationLexicon.load(path)
        logger.info("discovery: reusing lexicon with %d entries", len(lex))
    else:
        rng = np.random.default_rng(d.random_seed)
        params = _discovery_params(cfg)
        candidates = [
            (w, ix) for w, ix in sorted(by_word.items()) if len(ix) >= d.min_word_freq
        ]
        logger.info(
            "A3 discovery: %d word types seen, %d frequent enough to test (>= %d uses)",
            len(by_word), len(candidates), d.min_word_freq,
        )

        n_workers = d.n_workers if d.n_workers > 0 else (os.cpu_count() or 1)
        n_workers = max(1, min(n_workers, max(1, len(candidates))))

        # Largest words first: one with 600 occurrences costs far more than one
        # with 12, and starting those early stops a straggler from setting the
        # wall time.
        ordered = sorted(candidates, key=lambda wi: -len(wi[1]))
        entries: Dict[str, WordCodes] = {}

        if n_workers > 1 and len(ordered) > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            logger.info("A3 discovery: %d worker processes", n_workers)
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(
                        _discover_one,
                        _build_job(cfg, w, ix, occurrences, embeddings, rng, params),
                    ): w
                    for w, ix in ordered
                }
                bar = progress(
                    total=len(futures), desc="A3 discovering codes", unit="word"
                )
                for fut in as_completed(futures):
                    word = futures[fut]
                    try:
                        _, wc = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("discovery failed for %s: %s", word, exc)
                    else:
                        if wc is not None:
                            entries[word] = wc
                            bar.set_postfix(ambiguous=len(entries))
                    bar.update(1)
                bar.close()
        else:
            bar = progress(ordered, desc="A3 discovering codes", unit="word")
            for word, idxs in bar:
                _, wc = _discover_one(
                    _build_job(cfg, word, idxs, occurrences, embeddings, rng, params)
                )
                if wc is not None:
                    entries[word] = wc
                    bar.set_postfix(ambiguous=len(entries))

        lex = PronunciationLexicon(entries, d.max_codes_per_word)
        lex.save(path)
        logger.info(
            "discovery: %d word types examined, %d found ambiguous -> %s",
            len(by_word), len(lex.ambiguous_words), path,
        )

    # Assign every occurrence of every ambiguous word to a code.
    labels: Dict[Tuple[str, int], int] = {}
    amb_words = [w for w in by_word if lex.is_ambiguous(w)]
    for word in progress(amb_words, desc="A3 labelling occurrences", unit="word"):
        idxs = by_word[word]
        feats = embeddings[[occurrences[i].row for i in idxs]]
        codes = lex.assign(word, feats)
        for i, c in zip(idxs, codes):
            labels[(occurrences[i].uid, occurrences[i].word_index)] = int(c)
    logger.info("discovery: labelled %d ambiguous occurrences", len(labels))
    return lex, labels


# ---------------------------------------------------------------------------
# memmap helpers
# ---------------------------------------------------------------------------


class MemmapWriter:
    """Append-only memmap writer with a JSON sidecar describing the layout."""

    def __init__(self, path: Path, dim: Tuple[int, ...], dtype: str, capacity: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dim = tuple(dim)
        self.dtype = dtype
        self.capacity = capacity
        self.arr = np.lib.format.open_memmap(
            self.path, mode="w+", dtype=np.dtype(dtype), shape=(capacity, *self.dim)
        )
        self.n = 0

    def append(self, x: np.ndarray) -> int:
        if self.n >= self.capacity:
            raise RuntimeError(f"memmap {self.path} is full ({self.capacity} rows)")
        self.arr[self.n] = x
        self.n += 1
        return self.n - 1

    def close(self) -> None:
        self.arr.flush()
        meta = {"rows": self.n, "dim": list(self.dim), "dtype": self.dtype}
        with open(self.path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
