"""Run the full offline preprocessing pipeline.

    python scripts/preprocess.py --config configs/exp1_egyptian.yaml [--stage all]

Stages run in order and each is resumable: rerunning skips work whose output
already exists unless ``--force`` is given. Per-utterance artefacts are written
as ragged arrays (one concatenated buffer plus an offsets index), which keeps
random access O(1) in the dataloader without padding waste on disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from adaptts.data.preprocess import (  # noqa: E402
    CTCAlignerRunner,
    build_reading_lexicon,
    OccurrenceRecord,
    SpanEmbedder,
    Utterance,
    build_char_vocab,
    build_manifest,
    discover_pronunciation_codes,
)
from adaptts.text.normalize import tokenize_words  # noqa: E402
from adaptts.utils.console import enable_utf8_console  # noqa: E402
from adaptts.utils.logging_utils import (  # noqa: E402
    format_duration,
    log_config_summary,
    log_table,
    progress,
    setup_logging,
    stage,
)
from adaptts.utils.config import Config, load_config, save_config  # noqa: E402

enable_utf8_console()
setup_logging()
logger = logging.getLogger("preprocess")


# ---------------------------------------------------------------------------
# Ragged array storage
# ---------------------------------------------------------------------------


def save_ragged(path: Path, arrays: Sequence[np.ndarray], dtype: np.dtype) -> None:
    """Store variable-length arrays as one buffer plus offsets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lens = np.array([a.shape[0] for a in arrays], dtype=np.int64)
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    tail = arrays[0].shape[1:] if arrays and arrays[0].ndim > 1 else ()
    buf = np.lib.format.open_memmap(
        path, mode="w+", dtype=dtype, shape=(int(offsets[-1]), *tail)
    )
    for a, s in zip(arrays, offsets[:-1]):
        if a.shape[0]:
            buf[s : s + a.shape[0]] = a.astype(dtype, copy=False)
    buf.flush()
    np.save(path.with_name(path.stem + "_offsets.npy"), offsets)


class RaggedArray:
    """Read-only view over a ragged store written by :func:`save_ragged`."""

    def __init__(self, path: Path) -> None:
        self.buf = np.load(path, mmap_mode="r")
        self.offsets = np.load(Path(path).with_name(Path(path).stem + "_offsets.npy"))

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int) -> np.ndarray:
        return np.asarray(self.buf[self.offsets[i] : self.offsets[i + 1]])


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def load_audio(path: str, target_sr: int) -> torch.Tensor:
    """Load a mono waveform resampled to ``target_sr``."""
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(wav.mean(axis=1))
    if sr != target_sr:
        import torchaudio

        x = torchaudio.functional.resample(x, sr, target_sr)
    return x


# ---------------------------------------------------------------------------
# Stage A1: alignment
# ---------------------------------------------------------------------------


def stage_align(cfg: Config, utts: List[Utterance], device: torch.device, force: bool) -> Path:
    out = Path(cfg.paths.align_dir) / "word_alignments.jsonl"
    if out.exists() and not force:
        logger.info("A1 align: reusing %s", out)
        return out

    runner = CTCAlignerRunner(cfg, device)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_ok = n_drop = 0

    bar = progress(utts, desc="A1 forced alignment", unit="utt")
    with open(out, "w", encoding="utf-8") as f:
        for i, u in enumerate(bar):
            try:
                wav = load_audio(u.audio_path, runner.sample_rate)
                aligns, fps = runner.align(wav, u.text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("align failed for %s: %s", u.uid, exc)
                aligns, fps = [], 0.0

            kept = []
            for a in aligns:
                if fps <= 0:
                    continue
                t0s, t1s = a.to_seconds(fps)
                dur = t1s - t0s
                if dur < cfg.align.min_word_seconds or dur > cfg.align.max_word_seconds:
                    continue
                if a.score < cfg.align.score_threshold:
                    continue
                kept.append(
                    {
                        "word_index": a.word_index,
                        "word": a.word,
                        "start": round(float(t0s), 4),
                        "end": round(float(t1s), 4),
                        "score": round(float(a.score), 4),
                    }
                )
            n_ok += len(kept)
            n_drop += max(0, len(aligns) - len(kept))
            f.write(json.dumps({"uid": u.uid, "words": kept}, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                bar.set_postfix(spans=n_ok, dropped=n_drop)

    logger.info(
        "A1 alignment: kept %d word spans, dropped %d (%.1f%%) in %s",
        n_ok, n_drop, 100 * n_drop / max(n_ok + n_drop, 1), format_duration(time.time() - t0),
    )
    logger.info("           -> %s", out)
    return out


# ---------------------------------------------------------------------------
# Stage A2: span embeddings
# ---------------------------------------------------------------------------


def stage_span_embeddings(
    cfg: Config, utts: List[Utterance], align_path: Path, device: torch.device, force: bool
) -> Tuple[List[OccurrenceRecord], np.ndarray]:
    emb_path = Path(cfg.paths.spanemb_dir) / "span_embeddings.npy"
    occ_path = Path(cfg.paths.spanemb_dir) / "occurrences.jsonl"

    if emb_path.exists() and occ_path.exists() and not force:
        logger.info("A2 span embeddings: reusing %s", emb_path)
        embs = np.load(emb_path, mmap_mode="r")
        occ = [
            OccurrenceRecord(**json.loads(l))
            for l in open(occ_path, encoding="utf-8")
            if l.strip()
        ]
        return occ, np.asarray(embs)

    aligns: Dict[str, List[dict]] = {}
    with open(align_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            aligns[r["uid"]] = r["words"]

    embedder = SpanEmbedder(cfg, device)
    emb_path.parent.mkdir(parents=True, exist_ok=True)

    occurrences: List[OccurrenceRecord] = []
    chunks: List[np.ndarray] = []
    row = 0
    t0 = time.time()

    bar = progress(utts, desc="A2 span embeddings", unit="utt")
    for i, u in enumerate(bar):
        words = aligns.get(u.uid) or []
        if not words:
            continue
        try:
            wav = load_audio(u.audio_path, embedder.sample_rate)
            spans = [(w["start"], w["end"]) for w in words]
            vecs = embedder.embed(wav, spans)
        except Exception as exc:  # noqa: BLE001
            logger.warning("span embed failed for %s: %s", u.uid, exc)
            continue

        chunks.append(vecs.astype(np.float32))
        for w, _ in zip(words, range(len(vecs))):
            occurrences.append(
                OccurrenceRecord(
                    uid=u.uid, word_index=int(w["word_index"]), word=w["word"],
                    start_sec=float(w["start"]), end_sec=float(w["end"]),
                    score=float(w["score"]), row=row,
                )
            )
            row += 1

        if (i + 1) % 25 == 0:
            bar.set_postfix(spans=row)

    if not chunks:
        raise RuntimeError("no span embeddings produced; check stage A1 output")

    embs = np.concatenate(chunks, axis=0)
    np.save(emb_path, embs)
    with open(occ_path, "w", encoding="utf-8") as f:
        for o in occurrences:
            f.write(json.dumps(o.__dict__, ensure_ascii=False) + "\n")

    logger.info(
        "A2 span embeddings: %d spans x %d dims in %s",
        embs.shape[0], embs.shape[1], format_duration(time.time() - t0),
    )
    logger.info("           -> %s", emb_path)
    return occurrences, embs


# ---------------------------------------------------------------------------
# Stage A4: teacher cache
# ---------------------------------------------------------------------------


@torch.no_grad()
def stage_teacher(
    cfg: Config,
    utts: List[Utterance],
    lexicon,
    labels: Dict[Tuple[str, int], int],
    device: torch.device,
    force: bool,
) -> None:
    """Cache frozen MARBERTv2 word states, and fit the teacher's code head.

    The teacher head is a multinomial logistic regression over the teacher's
    contextual word vectors, fitted on the discovered codes. It is the source of
    the soft targets the student distills from. Fitting it here means the
    heavyweight model is never loaded during training.
    """
    out_dir = Path(cfg.paths.teacher_dir)
    hid_path = out_dir / "word_hidden.npy"
    head_path = out_dir / "teacher_head.npz"
    if hid_path.exists() and head_path.exists() and not force:
        logger.info("A4 teacher: reusing %s", out_dir)
        return

    from transformers import AutoModel, AutoTokenizer

    # use_fast is required: word_ids() below only exists on fast tokenizers.
    tok = AutoTokenizer.from_pretrained(cfg.teacher.model_id, use_fast=True)
    model = AutoModel.from_pretrained(cfg.teacher.model_id).to(device).eval()
    if cfg.teacher.dtype == "float16" and device.type == "cuda":
        model = model.half()

    out_dir.mkdir(parents=True, exist_ok=True)
    layers = tuple(cfg.teacher.layers)
    bs = cfg.teacher.batch_size

    all_word_vecs: List[np.ndarray] = []
    uid_order: List[str] = []
    t0 = time.time()

    n_batches = (len(utts) + bs - 1) // bs
    bar = progress(range(0, len(utts), bs), desc="A4 teacher cache",
                   unit="batch", total=n_batches)
    for start in bar:
        batch = utts[start : start + bs]
        texts = [u.text for u in batch]
        word_lists = [tokenize_words(t)[0] for t in texts]

        enc = tok(
            [w if w else [""] for w in word_lists],
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.teacher.max_length,
        )
        # Capture word_ids BEFORE moving to device: converting the BatchEncoding
        # to a plain dict drops the fast-tokenizer methods, and a silent None
        # here would produce all-zero teacher vectors.
        word_ids_per_item = [enc_word_ids(tok, enc, bi) for bi in range(len(batch))]

        model_inputs = {k: v.to(device) for k, v in enc.items()}
        out = model(**model_inputs, output_hidden_states=True)
        hs = out.hidden_states
        feats = torch.stack([hs[l] for l in layers], dim=0).mean(0).float()  # [B, L, H]

        for bi, u in enumerate(batch):
            wid = word_ids_per_item[bi]
            n_words = len(word_lists[bi])
            H = feats.shape[-1]
            acc = torch.zeros(n_words, H, device=device)
            cnt = torch.zeros(n_words, device=device)
            for ti, w in enumerate(wid):
                if w is None or w >= n_words:
                    continue
                acc[w] += feats[bi, ti]
                cnt[w] += 1
            vec = (acc / cnt.clamp(min=1).unsqueeze(-1)).cpu().numpy().astype(np.float16)
            all_word_vecs.append(vec)
            uid_order.append(u.uid)

        if start and (start // bs) % 10 == 0:
            bar.set_postfix(utts=start + len(batch))

    save_ragged(hid_path, all_word_vecs, np.dtype(np.float16))
    with open(out_dir / "uid_order.json", "w", encoding="utf-8") as f:
        json.dump(uid_order, f)

    fit_teacher_head(cfg, utts, uid_order, hid_path, lexicon, labels, head_path)
    logger.info("A4 teacher: cached word states -> %s", hid_path)


def enc_word_ids(tok, enc, batch_index: int):
    """Return per-token word ids for one batch item.

    This mapping is what aligns subword tokens back to whole words, so a silent
    failure here would yield all-zero teacher vectors and a quietly useless
    distillation signal. We therefore raise rather than degrade.
    """
    getter = getattr(enc, "word_ids", None)
    if getter is None:
        raise RuntimeError(
            f"tokenizer {tok.__class__.__name__} did not return a BatchEncoding with "
            "word_ids(). A fast tokenizer is required; install `tokenizers` or pick a "
            "model whose tokenizer has a fast implementation."
        )
    ids = getter(batch_index)
    if ids is None:
        raise RuntimeError(
            "word_ids() returned None, which means a slow tokenizer is in use. "
            "Load the teacher with use_fast=True."
        )
    return ids


def fit_teacher_head(
    cfg: Config,
    utts: List[Utterance],
    uid_order: List[str],
    hid_path: Path,
    lexicon,
    labels: Dict[Tuple[str, int], int],
    out_path: Path,
) -> None:
    """Fit a logistic-regression code head on the teacher's word vectors.

    One shared head over all ambiguous words, with a per-word legality mask,
    mirrors the student's design so the distillation targets are comparable.
    """
    ragged = RaggedArray(hid_path)
    index = {uid: i for i, uid in enumerate(uid_order)}

    X: List[np.ndarray] = []
    y: List[int] = []
    for (uid, widx), code in labels.items():
        i = index.get(uid)
        if i is None:
            continue
        vecs = ragged[i]
        if widx >= vecs.shape[0]:
            continue
        X.append(vecs[widx].astype(np.float32))
        y.append(int(code))

    if len(X) < 32:
        logger.warning(
            "A4 teacher: only %d labelled examples; skipping teacher head "
            "(the student will train on hard labels alone)", len(X)
        )
        np.savez(out_path, empty=np.array([1]))
        return

    Xa = np.stack(X)
    ya = np.array(y, dtype=np.int64)
    K = cfg.discovery.max_codes_per_word

    # Standardize, then fit multinomial logistic regression by full-batch
    # gradient descent on the GPU. No sklearn dependency, a few seconds.
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    Z = torch.from_numpy((Xa - mu) / sd)
    t = torch.from_numpy(ya)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Z, t = Z.to(dev), t.to(dev)

    W = torch.zeros(Z.shape[1], K, device=dev, requires_grad=True)
    b = torch.zeros(K, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Z @ W + b, t) + 1e-4 * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        acc = float(((Z @ W + b).argmax(-1) == t).float().mean())
    logger.info("A4 teacher head: fitted on %d examples, train acc %.3f", len(y), acc)

    np.savez(
        out_path,
        W=W.detach().cpu().numpy(), b=b.detach().cpu().numpy(),
        mu=mu, sd=sd, acc=np.array([acc]),
    )


# ---------------------------------------------------------------------------
# Stage A5: Mimi codec encoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def stage_codec(cfg: Config, utts: List[Utterance], device: torch.device, force: bool) -> None:
    codes_path = Path(cfg.paths.codes_dir) / "codes.npy"
    if codes_path.exists() and not force:
        logger.info("A5 codec: reusing %s", codes_path)
        return

    from transformers import MimiModel

    model = MimiModel.from_pretrained(cfg.codec_model_id).to(device).eval()
    sr = model.config.sampling_rate
    if sr != cfg.audio.codec_sample_rate:
        logger.warning(
            "codec sample rate %d differs from config %d; using the model's",
            sr, cfg.audio.codec_sample_rate,
        )

    codes_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: List[np.ndarray] = []
    uid_order: List[str] = []
    t0 = time.time()

    bar = progress(utts, desc="A5 codec encoding", unit="utt")
    for i, u in enumerate(bar):
        try:
            wav = load_audio(u.audio_path, sr).to(device)
            enc = model.encode(
                wav[None, None], num_quantizers=cfg.audio.n_quantizers
            )
            c = enc.audio_codes[0].transpose(0, 1).cpu().numpy().astype(np.int16)  # [T, Q]
        except Exception as exc:  # noqa: BLE001
            logger.warning("codec failed for %s: %s", u.uid, exc)
            continue
        if c.shape[1] != cfg.audio.n_quantizers:
            raise RuntimeError(
                f"codec returned {c.shape[1]} quantizers, config says "
                f"{cfg.audio.n_quantizers}"
            )
        arrays.append(c)
        uid_order.append(u.uid)

        if (i + 1) % 25 == 0:
            bar.set_postfix(frames=sum(a.shape[0] for a in arrays[-25:]))

    if not arrays:
        raise RuntimeError("codec produced no output")
    save_ragged(codes_path, arrays, np.dtype(np.int16))
    with open(Path(cfg.paths.codes_dir) / "uid_order.json", "w", encoding="utf-8") as f:
        json.dump(uid_order, f)
    total_frames = sum(a.shape[0] for a in arrays)
    logger.info(
        "A5 codec: %d utterances, %d frames (%.1f min of audio) in %s",
        len(arrays), total_frames, total_frames / 12.5 / 60,
        format_duration(time.time() - t0),
    )
    logger.info("           -> %s", codes_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="AdapTTS offline preprocessing")
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--stage", default="all",
        choices=["all", "manifest", "align", "spanemb", "discover", "teacher", "codec"],
    )
    ap.add_argument(
        "--force", action="store_true",
        help="recompute the stage named by --stage, even if it is cached",
    )
    ap.add_argument(
        "--force-upstream", action="store_true",
        help="also recompute the stages the named one depends on "
             "(use when an upstream cache is stale, not merely to rerun)",
    )
    ap.add_argument("--limit", type=int, default=0, help="debug: only N utterances")
    args = ap.parse_args()

    t_start = time.time()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("AdapTTS preprocessing")
    logger.info("device: %s", device)
    if args.force and args.stage != "all" and not args.force_upstream:
        logger.info(
            "--force applies to stage %r only; upstream caches are reused "
            "(pass --force-upstream to rebuild those too)", args.stage,
        )
    log_config_summary(logger, cfg)

    def run(stage: str) -> bool:
        return args.stage in ("all", stage)

    def force_for(stage: str) -> bool:
        """Should this stage recompute?

        --force applies to the stage the user named. It does not cascade to the
        stages that one depends on, because those caches are usually valid and
        recomputing them is expensive: span embeddings alone take 11 minutes on
        the 68-hour corpus. --force-upstream opts back in.
        """
        if not args.force:
            return False
        if args.stage == "all" or args.force_upstream:
            return True
        return args.stage == stage


    Path(cfg.paths.cache_dir).mkdir(parents=True, exist_ok=True)
    save_config(cfg, Path(cfg.paths.cache_dir) / "config_snapshot.yaml")

    utts = build_manifest(cfg, force=force_for("manifest"))
    if args.limit:
        utts = utts[: args.limit]
        logger.warning("limiting to %d utterances (debug)", len(utts))
    build_char_vocab(cfg, utts, force=force_for("manifest"))

    align_path = Path(cfg.paths.align_dir) / "word_alignments.jsonl"
    if run("align"):
        align_path = stage_align(cfg, utts, device, force_for("align"))
    elif not align_path.exists() and args.stage in ("spanemb", "discover"):
        raise SystemExit("stage 'align' must run before 'spanemb'/'discover'")

    if run("discover"):
        # Labels come from the diacritizer, not from acoustic clustering.
        # Clustering returned the channel's subscribe pitch as "homographs"
        # while the real ones got a single code; see text/diacritics.py.
        lexicon, labels = build_reading_lexicon(cfg, force=force_for("discover"))
        report_readings(cfg, lexicon)
    else:
        from adaptts.text.diacritics import ReadingLexicon

        lexicon = (
            ReadingLexicon.load(Path(cfg.paths.reading_lexicon_path))
            if Path(cfg.paths.reading_lexicon_path).exists() else None
        )
        labels = {}
        lp = Path(cfg.paths.cache_dir) / "code_labels.json"
        if lp.exists():
            raw = json.load(open(lp, encoding="utf-8"))
            labels = {(k.split("\t")[0], int(k.split("\t")[1])): v for k, v in raw.items()}

    if run("teacher"):
        if lexicon is None:
            raise SystemExit("stage 'discover' must run before 'teacher'")
        stage_teacher(cfg, utts, lexicon, labels, device, force_for("teacher"))

    if run("codec"):
        stage_codec(cfg, utts, device, force_for("codec"))

    logger.info("")
    logger.info("=" * 68)
    logger.info("PREPROCESSING COMPLETE in %s", format_duration(time.time() - t_start))
    logger.info("cache: %s", cfg.paths.cache_dir)
    logger.info("=" * 68)


def report_readings(cfg: Config, lexicon) -> None:
    """Write a human-readable report of the readings found.

    This is the decision point for the whole project: if the words listed here
    are real homographs rather than register artifacts, the labels are sound.
    """
    words = lexicon.ambiguous_words
    path = Path(cfg.paths.cache_dir) / "reading_report.txt"
    rows = sorted((lexicon.entries[w] for w in words), key=lambda e: -e.total)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(lexicon)} word types, {len(words)} with several readings\n\n")
        f.write(f"{'word':<18}{'readings':>9}{'uses':>7}  counts / examples\n")
        f.write("-" * 78 + "\n")
        for e in rows:
            ex = "  ".join(f"{x}({n})" for x, n in zip(e.examples, e.counts))
            f.write(f"{e.word:<18}{e.n_codes:>9}{e.total:>7}  {ex}\n")

    logger.info("reading report -> %s", path)
    logger.info("%d word types have more than one reading", len(words))
    for e in rows[:15]:
        ex = "  ".join(f"{x}({n})" for x, n in zip(e.examples, e.counts))
        logger.info("  %s: %d readings  %s", e.word, e.n_codes, ex)


if __name__ == "__main__":
    main()
