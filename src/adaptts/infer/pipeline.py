"""Inference pipeline: inspectable, controllable, adaptive.

The design requirement is that a user can see which words the system found hard,
see what reading it chose, and override that reading **without writing
diacritics**. That is possible because the pronunciation decision is a discrete
code, not a hidden activation, so overriding it is a legal in-distribution edit.

Typical use::

    tts = AdapTTS.from_checkpoints(config, ctx_ckpt, acoustic_ckpt)
    plan = tts.analyze("انا شوفت علم مصر بيرفرف")
    print(plan)                       # per-word codes, confidence, difficulty
    plan.set_code("علم", 0)           # override a reading by index
    wav = tts.synthesize(plan)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..text.diacritics import ReadingLexicon
from ..models.acoustic import AcousticModel
from ..models.context_encoder import ContextEncoder
from ..text.normalize import normalize_text, tokenize_words
from ..text.vocab import CharVocab
from ..utils.config import Config, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------


def _first(*values: Any) -> Any:
    """The first value that is not None.

    Gives control knobs a consistent precedence: an explicit call argument, then
    whatever the plan carries, then the config default.
    """
    for v in values:
        if v is not None:
            return v
    return None


def split_user_diacritics(
    text: str, lexicon: Optional["ReadingLexicon"]
) -> Tuple[str, Dict[int, int], Dict[int, str]]:
    """Separate a user's optional diacritics from the text the model reads.

    This is the override channel. The system never requires diacritized input,
    but a user who hears a wrong reading can fix it by writing the vowels on
    that one word:

        انا كنت مصر على ان مصر عندها امكانيات        -> the model decides both
        انا كنت مُصِرّ على ان مصر عندها امكانيات      -> first pinned, second free

    That is a better interface than passing a code number, because codes are
    assigned by corpus frequency: nobody can know that code 1 means مُصِرّ, and
    the answer changes when the lexicon is rebuilt.

    Returns ``(bare_text, overrides, unresolved, variants)``:

    * ``bare_text`` has every diacritic removed, so the model's input
      distribution is exactly what it was trained on. Diacritics never reach
      the model as characters.
    * ``overrides`` maps word index -> code for marks that resolved.
    * ``unresolved`` maps word index -> the reason it did not, so the caller can
      say so instead of silently mispronouncing.
    * ``variants`` is one phoneme-variant code per character of ``bare_text``,
      carrying a ``~`` on ق/ج/ف or a sounded final /t/ through to the model.
    """
    from ..text.diacritics import letter_marks, strip_diacritics

    from ..text.diacritics import letter_variants, normalize_variant_marks

    overrides: Dict[int, int] = {}
    unresolved: Dict[int, str] = {}
    bare_words: List[str] = []
    variant_parts: List[List[int]] = []

    for i, token in enumerate(text.split()):
        bare = strip_diacritics(token)
        bare_words.append(bare)
        # Phoneme variants the user asked for: a ~ on ق/ج/ف, or sukun on a final
        # ة to sound the /t/. Kept even when the reading itself is left to the
        # model, since the two are independent.
        v = letter_variants(normalize_variant_marks(token))
        variant_parts.append(v if len(v) == len(bare) else [0] * len(bare))
        # Did the user actually mark anything on this word?
        if not any(mk for _, mk in letter_marks(token)):
            continue
        if lexicon is None:
            unresolved[i] = "no lexicon loaded"
            continue
        code, reason = lexicon.code_from_partial_marks(token)
        if code >= 0:
            # A single-reading word needs no override; recording one would
            # imply a choice that does not exist.
            if lexicon.n_codes(bare) > 1:
                overrides[i] = code
        else:
            unresolved[i] = reason

    # Flatten to one code per character of the joined text, including the spaces.
    flat: List[int] = []
    for j, part in enumerate(variant_parts):
        if j:
            flat.append(0)          # the separating space
        flat.extend(part)
    return " ".join(bare_words), overrides, unresolved, flat


def prepare_inputs(
    text: str,
    vocab: CharVocab,
    lexicon: Optional["ReadingLexicon"],
    device: torch.device,
    cfg: Optional[Config] = None,
) -> Dict[str, torch.Tensor]:
    """Normalize and tensorize one sentence for batch size 1.

    Any diacritics in ``text`` are read as pronunciation overrides and then
    removed, so what reaches the model is always undiacritized.
    """
    # Before normalization, which would discard the marks entirely.
    text, user_overrides, unresolved, variant_codes = split_user_diacritics(
        text, lexicon
    )
    pre_norm_len = len(text)

    if cfg is not None:
        text = normalize_text(
            text,
            strip_diacritics_flag=cfg.text.strip_diacritics,
            lowercase_latin=cfg.text.lowercase_latin,
            normalize_alef=cfg.text.normalize_alef,
            normalize_digits=cfg.text.normalize_digits,
        )
    else:
        text = normalize_text(text)

    words, spans = tokenize_words(text)
    if not words:
        raise ValueError("the input contains no pronounceable words")
    ids, widx = vocab.encode_with_word_index(text, spans)

    n_codes = [
        lexicon.n_codes(w) if lexicon is not None else 1 for w in words
    ]

    # Variants are indexed by character of the pre-normalization text. If
    # normalization changed the length (a digit expanded to words, say) the
    # mapping no longer holds, so drop them rather than attach a /v/ to the
    # wrong letter. char_ids carries BOS and EOS, hence the offset of one.
    variants = torch.zeros(1, len(ids), dtype=torch.long, device=device)
    if len(text) == pre_norm_len and len(variant_codes) == len(text):
        body = torch.tensor(variant_codes, dtype=torch.long, device=device)
        variants[0, 1 : 1 + body.numel()] = body

    return {
        "text": text,
        "words": words,
        "user_overrides": user_overrides,
        "unresolved_marks": unresolved,
        "variants": variants,
        "char_ids": torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0),
        "word_index": torch.tensor(widx, dtype=torch.long, device=device).unsqueeze(0),
        "n_codes": torch.tensor(n_codes, dtype=torch.long, device=device).unsqueeze(0),
        "char_padding_mask": torch.zeros(1, len(ids), dtype=torch.bool, device=device),
    }


# ---------------------------------------------------------------------------
# The user-facing plan object
# ---------------------------------------------------------------------------


@dataclass
class WordPlan:
    index: int
    word: str
    n_codes: int
    code: int
    probs: List[float]
    difficulty: float
    # True when this reading came from diacritics the user typed rather than
    # from the model, so the plan can show which decisions are the model's.
    from_user: bool = False

    @property
    def is_ambiguous(self) -> bool:
        return self.n_codes > 1

    @property
    def confidence(self) -> float:
        return float(self.probs[self.code]) if self.probs else 1.0


@dataclass
class SynthesisPlan:
    """What the model decided, in a form a human can read and change."""

    text: str
    words: List[WordPlan]
    sentence_difficulty: float
    depth: int
    inputs: Dict[str, torch.Tensor] = field(repr=False, default_factory=dict)
    overrides: Dict[int, int] = field(default_factory=dict)
    # Words where the user wrote diacritics that could not be resolved, with the
    # reason. Reported rather than silently ignored.
    unresolved_marks: Dict[int, str] = field(default_factory=dict)
    # Per-sentence generation controls. None means "use the config default".
    tempo: Optional[float] = None
    cfg_scale: Optional[float] = None
    temperature: Optional[float] = None

    # -- inspection ----------------------------------------------------

    @property
    def hard_words(self) -> List[WordPlan]:
        return [w for w in self.words if w.is_ambiguous]

    def complexity_table(self) -> List[Dict[str, Any]]:
        """Per-word compute and difficulty, as rows ready to print or plot.

        This is where the adaptive claim is falsifiable. A word with one known
        reading should be cheap and a homograph should not; if they cost the
        same, the difficulty head has learned nothing and the adaptive-depth
        story is empty. Exposing it makes that visible rather than asserted.

        ``depth`` is the depth this word's difficulty alone would select, which
        is not necessarily the sentence's depth: generation runs at one depth
        for the whole sentence, and the per-word number shows what is driving
        it.
        """
        rows: List[Dict[str, Any]] = []
        for w in self.words:
            rows.append({
                "index": w.index,
                "word": w.word,
                "readings": w.n_codes,
                "code": w.code if w.is_ambiguous else None,
                "confidence": w.confidence if w.is_ambiguous else None,
                "difficulty": w.difficulty,
                "depth": self._depth_for(w.difficulty),
                "source": ("user" if w.from_user
                           else "model" if w.is_ambiguous else "unambiguous"),
            })
        return rows

    def _depth_for(self, difficulty: float) -> Optional[int]:
        """Which exit this difficulty maps to, if the thresholds are known."""
        thr = getattr(self, "_depth_thresholds", None)
        if not thr:
            return None
        exits, tau_low, tau_high = thr
        if difficulty <= tau_low:
            return exits[0]
        if difficulty <= tau_high:
            return exits[min(1, len(exits) - 1)]
        return exits[-1]

    def complexity_report(self) -> str:
        """The complexity table as text, hardest words first."""
        rows = self.complexity_table()
        lines = [
            f"text: {self.text}",
            f"sentence difficulty {self.sentence_difficulty:.3f} -> depth {self.depth}",
            "",
            f"{'#':>3}  {'word':<16}{'read':>5}{'diff':>8}{'depth':>7}"
            f"{'conf':>8}  source",
            "-" * 62,
        ]
        for r in sorted(rows, key=lambda r: -r["difficulty"]):
            conf = f"{r['confidence']:.3f}" if r["confidence"] is not None else "-"
            depth = r["depth"] if r["depth"] is not None else "-"
            lines.append(
                f"{r['index']:>3}  {r['word']:<16}{r['readings']:>5}"
                f"{r['difficulty']:>8.3f}{str(depth):>7}{conf:>8}  {r['source']}"
            )
        if self.unresolved_marks:
            lines.append("")
            lines.append("diacritics that could not be applied:")
            for i, why in sorted(self.unresolved_marks.items()):
                word = self.words[i].word if 0 <= i < len(self.words) else f"#{i}"
                lines.append(f"  {word}: {why}")
        return "\n".join(lines)

    def __str__(self) -> str:
        lines = [
            f'text: {self.text}',
            f"sentence difficulty: {self.sentence_difficulty:.3f}   depth: {self.depth}",
        ]
        hard = self.hard_words
        if not hard:
            lines.append("no ambiguous words: every word has a single known reading")
            return "\n".join(lines)
        lines.append("")
        lines.append(f"{'#':>3}  {'word':<16}{'readings':>9}{'chosen':>8}"
                     f"{'confidence':>12}{'difficulty':>12}")
        lines.append("-" * 64)
        for w in hard:
            mark = " *" if w.index in self.overrides else ""
            lines.append(
                f"{w.index:>3}  {w.word:<16}{w.n_codes:>9}{w.code:>8}"
                f"{w.confidence:>12.3f}{w.difficulty:>12.3f}{mark}"
            )
        if self.overrides:
            lines.append("\n* = manually overridden")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "sentence_difficulty": self.sentence_difficulty,
            "depth": self.depth,
            "words": [
                {
                    "index": w.index, "word": w.word, "n_codes": w.n_codes,
                    "code": w.code, "confidence": w.confidence,
                    "difficulty": w.difficulty, "probs": w.probs,
                }
                for w in self.words
            ],
            "overrides": self.overrides,
        }

    # -- control -------------------------------------------------------

    def set_code(self, word_or_index: Any, code: int, occurrence: int = -1) -> "SynthesisPlan":
        """Override the chosen reading.

        ``word_or_index`` may be a word index, or the word itself. When a word
        occurs more than once, ``occurrence`` selects which one (-1 = all).
        """
        if isinstance(word_or_index, int):
            targets = [word_or_index]
        else:
            matches = [w.index for w in self.words if w.word == word_or_index]
            if not matches:
                raise KeyError(f"{word_or_index!r} is not in this sentence")
            targets = matches if occurrence < 0 else [matches[occurrence]]

        for i in targets:
            if not 0 <= i < len(self.words):
                raise IndexError(f"word index {i} out of range")
            w = self.words[i]
            if not 0 <= code < w.n_codes:
                raise ValueError(
                    f"{w.word!r} has {w.n_codes} reading(s); valid codes are "
                    f"0..{w.n_codes - 1}, got {code}"
                )
            w.code = code
            self.overrides[i] = code
        return self

    def set_reading(self, word: str, diacritized: str) -> "SynthesisPlan":
        """Pin a reading by writing the vowels, not by picking a code number.

            plan.set_reading("مصر", "مُصِرّ")

        The same channel as typing diacritics in the input text, available on an
        existing plan so a correction does not require re-analysis. Raises with
        the reason when the marks cannot be resolved, rather than guessing.
        """
        lex = getattr(self, "_lexicon", None)
        if lex is None:
            raise RuntimeError("this plan has no lexicon attached")
        code, reason = lex.code_from_partial_marks(diacritized)
        if code < 0:
            raise ValueError(f"cannot resolve {diacritized!r}: {reason}")
        return self.set_code(word, code)

    def set_depth(self, depth: int) -> "SynthesisPlan":
        """Fix the generation depth, overriding the adaptive choice."""
        self.depth = depth
        return self

    def set_budget(self, budget: float) -> "SynthesisPlan":
        """Cap compute as a fraction of the deepest exit, in [0, 1].

        A budget does not force a depth; it sets a ceiling. An easy sentence
        still runs shallow, so this trades quality for speed only where the
        model actually wanted the compute.
        """
        if not 0.0 < budget <= 1.0:
            raise ValueError(f"budget must be in (0, 1], got {budget}")
        exits = getattr(self, "_exit_layers", None)
        if not exits:
            raise RuntimeError("this plan has no exit layers attached")
        ceiling = max(exits[0], int(round(max(exits) * budget)))
        allowed = [e for e in exits if e <= ceiling] or [exits[0]]
        self.depth = min(self.depth, max(allowed))
        return self

    def set_tempo(self, tempo: float) -> "SynthesisPlan":
        """Speaking rate, where 1.0 is the model's own pace.

        Above 1.0 is faster, below is slower. Applied by scaling the duration
        the model predicts, so prosody is preserved rather than resampled.
        """
        if not 0.25 <= tempo <= 4.0:
            raise ValueError(f"tempo must be in [0.25, 4.0], got {tempo}")
        self.tempo = tempo
        return self

    def set_cfg_scale(self, scale: float) -> "SynthesisPlan":
        """How strongly to follow the conditioning, including a reference voice.

        1.0 disables guidance. Higher values track the reference more closely at
        some cost in naturalness.
        """
        if not 0.0 <= scale <= 10.0:
            raise ValueError(f"cfg_scale must be in [0, 10], got {scale}")
        self.cfg_scale = scale
        return self

    def set_temperature(self, t: float) -> "SynthesisPlan":
        if not 0.0 <= t <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {t}")
        self.temperature = t
        return self

    def pc_per_char(self, max_codes: int) -> torch.Tensor:
        """Build the per-character code tensor the acoustic model consumes."""
        widx = self.inputs["word_index"]
        pc = torch.full_like(widx, max_codes)
        for w in self.words:
            if w.is_ambiguous:
                pc[widx == w.index] = w.code
        return pc


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


class AdapTTS:
    def __init__(
        self,
        cfg: Config,
        vocab: CharVocab,
        lexicon: Optional["ReadingLexicon"],
        context_encoder: Optional[ContextEncoder],
        acoustic: Optional[AcousticModel],
        codec=None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfg = cfg
        self.vocab = vocab
        self.lexicon = lexicon
        self.context_encoder = context_encoder
        self.acoustic = acoustic
        self.codec = codec
        self.device = device or torch.device("cpu")
        for m in (self.context_encoder, self.acoustic, self.codec):
            if m is not None:
                m.eval()

    # -- construction --------------------------------------------------

    @classmethod
    def from_checkpoints(
        cls,
        config_path: str,
        context_ckpt: Optional[str] = None,
        acoustic_ckpt: Optional[str] = None,
        device: str = "cpu",
        load_codec: bool = True,
    ) -> "AdapTTS":
        cfg = load_config(config_path)
        dev = torch.device(device)
        vocab = CharVocab.load(Path(cfg.paths.charvocab_path))
        lex_path = Path(cfg.paths.reading_lexicon_path)
        lexicon = ReadingLexicon.load(lex_path) if lex_path.exists() else None

        ctx = None
        cpath = context_ckpt or str(Path(cfg.paths.ckpt_dir) / "context_encoder" / "best.pt")
        if Path(cpath).exists():
            ctx = ContextEncoder(
                vocab_size=len(vocab), d_model=cfg.context_encoder.d_model,
                n_layers=cfg.context_encoder.n_layers, n_heads=cfg.context_encoder.n_heads,
                d_ff=cfg.context_encoder.d_ff, dropout=0.0,
                max_codes=cfg.context_encoder.max_codes,
                teacher_dim=cfg.teacher.hidden_size, pad_id=vocab.pad_id,
            )
            sd = torch.load(cpath, map_location="cpu", weights_only=False)
            ctx.load_state_dict(sd["model"])
            ctx = ctx.to(dev)

        ac = None
        apath = acoustic_ckpt or str(Path(cfg.paths.ckpt_dir) / "acoustic" / "best.pt")
        if Path(apath).exists():
            ac = AcousticModel(
                vocab_size=len(vocab), n_quantizers=cfg.audio.n_quantizers,
                codebook_size=cfg.audio.codebook_size, d_model=cfg.acoustic.d_model,
                n_layers=cfg.acoustic.n_layers, n_heads=cfg.acoustic.n_heads,
                d_ff=cfg.acoustic.d_ff, dropout=0.0,
                text_d_model=cfg.acoustic.text_d_model,
                text_n_layers=cfg.acoustic.text_n_layers,
                text_n_heads=cfg.acoustic.text_n_heads,
                depth_d_model=cfg.acoustic.depth_d_model,
                depth_n_layers=cfg.acoustic.depth_n_layers,
                depth_n_heads=cfg.acoustic.depth_n_heads,
                speaker_dim=cfg.acoustic.speaker_dim,
                max_codes=cfg.discovery.max_codes_per_word,
                pc_embed_dim=cfg.acoustic.pc_embed_dim,
                exit_layers=cfg.acoustic.exit_layers, pad_id=vocab.pad_id,
            )
            sd = torch.load(apath, map_location="cpu", weights_only=False)
            ac.load_state_dict(sd["model"])
            ac = ac.to(dev)

        codec = None
        if load_codec and ac is not None:
            from transformers import MimiModel

            codec = MimiModel.from_pretrained(cfg.codec_model_id).to(dev).eval()

        return cls(cfg, vocab, lexicon, ctx, ac, codec, dev)

    # -- stage 1: analyze ----------------------------------------------

    @torch.no_grad()
    def analyze(self, text: str, depth: Optional[int] = None) -> SynthesisPlan:
        """Resolve every ambiguous word and report the decision."""
        inp = prepare_inputs(text, self.vocab, self.lexicon, self.device, self.cfg)
        words = inp["words"]
        n_codes = inp["n_codes"]

        if self.context_encoder is None:
            plans = [
                WordPlan(i, w, int(n_codes[0, i]), 0, [1.0], 0.0)
                for i, w in enumerate(words)
            ]
            plan = SynthesisPlan(inp["text"], plans, 0.0,
                                 depth or self.cfg.acoustic.n_layers, inp)
            plan._lexicon = self.lexicon
            plan._exit_layers = list(self.cfg.acoustic.exit_layers)
            # Without a context encoder there is no difficulty signal, so no
            # thresholds either; complexity_table reports depth as unknown.
            plan.unresolved_marks = dict(inp.get("unresolved_marks") or {})
            for i, code in (inp.get("user_overrides") or {}).items():
                if 0 <= i < len(plan.words) and 0 <= code < plan.words[i].n_codes:
                    plan.words[i].code = code
                    plan.overrides[i] = code
                    plan.words[i].from_user = True
            return plan

        out = self.context_encoder(
            inp["char_ids"], inp["word_index"], n_codes, inp["char_padding_mask"]
        )
        codes = out.code_logits.argmax(-1)[0]
        probs = out.code_probs[0]
        diff = out.difficulty[0]

        plans: List[WordPlan] = []
        for i, w in enumerate(words):
            k = int(n_codes[0, i])
            plans.append(
                WordPlan(
                    index=i, word=w, n_codes=k, code=int(codes[i]),
                    probs=[float(p) for p in probs[i, :k]],
                    difficulty=float(diff[i]),
                )
            )

        word_mask = torch.ones_like(n_codes, dtype=torch.bool)
        sent_diff = float(self.context_encoder.sentence_difficulty(out.difficulty, word_mask)[0])
        chosen = depth if depth else self.select_depth(sent_diff)
        plan = SynthesisPlan(inp["text"], plans, sent_diff, chosen, inp)
        # Attached so the plan can resolve later corrections and report the
        # depth each word's difficulty would select.
        plan._lexicon = self.lexicon
        plan._exit_layers = list(self.cfg.acoustic.exit_layers)
        plan._depth_thresholds = (
            list(self.cfg.acoustic.exit_layers),
            self.cfg.inference.tau_low,
            self.cfg.inference.tau_high,
        )

        # Apply any reading the user pinned by writing diacritics. This happens
        # after the model has run, so the plan still reports what the model
        # would have chosen on its own and the override is visible as such.
        plan.unresolved_marks = dict(inp.get("unresolved_marks") or {})
        for i, code in (inp.get("user_overrides") or {}).items():
            if 0 <= i < len(plan.words) and 0 <= code < plan.words[i].n_codes:
                plan.words[i].code = code
                plan.overrides[i] = code
                plan.words[i].from_user = True
            else:
                # Token count changed in normalization, so the index no longer
                # refers to the same word. Better to report than to pin the
                # wrong one.
                plan.unresolved_marks[i] = (
                    "word position shifted during normalization"
                )
        return plan

    def select_depth(self, difficulty: float) -> int:
        """Map sentence difficulty onto one of the trained exit depths.

        This is the adaptive-compute decision: an easy sentence runs a shallow
        exit, a hard one runs the full stack.
        """
        ic = self.cfg.inference
        if ic.force_depth:
            return ic.force_depth
        exits = list(self.cfg.acoustic.exit_layers)
        if difficulty <= ic.tau_low:
            return exits[0]
        if difficulty <= ic.tau_high:
            return exits[min(1, len(exits) - 1)]
        return exits[-1]

    # -- stage 2: synthesize -------------------------------------------

    @torch.no_grad()
    def synthesize(
        self,
        plan_or_text: Any,
        speaker: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        cfg_scale: Optional[float] = None,
        seed: Optional[int] = None,
        tempo: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Render a plan (or a raw string) to a waveform.

        Explicit arguments win over values set on the plan, which in turn win
        over the config defaults, so a caller can override one knob without
        restating the others.
        """
        if self.acoustic is None or self.codec is None:
            raise RuntimeError(
                "synthesis needs both an acoustic checkpoint and the codec; "
                "this instance was loaded without them"
            )
        plan = plan_or_text if isinstance(plan_or_text, SynthesisPlan) else self.analyze(plan_or_text)
        ic = self.cfg.inference

        gen = None
        s = seed if seed is not None else ic.seed
        if s:
            gen = torch.Generator(device=self.device).manual_seed(int(s))

        if speaker is None:
            speaker = torch.zeros(1, self.cfg.acoustic.speaker_dim, device=self.device)

        pc = plan.pc_per_char(self.cfg.discovery.max_codes_per_word)
        t0 = time.perf_counter()
        codes, stats = self.acoustic.generate(
            plan.inputs["char_ids"], pc, speaker, plan.inputs["char_padding_mask"],
            depth=plan.depth,
            temperature=_first(temperature, plan.temperature, ic.temperature),
            top_k=ic.top_k, top_p=ic.top_p,
            cfg_scale=_first(cfg_scale, plan.cfg_scale, ic.cfg_scale),
            tempo=_first(tempo, plan.tempo, 1.0),
            variants=plan.inputs.get("variants"),
            monotonic_strength=self.cfg.acoustic.monotonic_prior_weight,
            repetition_window=ic.repetition_window,
            repetition_max_repeats=ic.repetition_max_repeats,
            generator=gen,
        )
        t_lm = time.perf_counter() - t0

        wav = self.codec.decode(codes.transpose(1, 2)).audio_values[0, 0].float().cpu().numpy()
        total = time.perf_counter() - t0
        sr = self.codec.config.sampling_rate
        audio_sec = len(wav) / sr

        return wav, {
            "sample_rate": sr,
            "audio_seconds": audio_sec,
            "lm_seconds": t_lm,
            "total_seconds": total,
            "real_time_factor": total / max(audio_sec, 1e-6),
            "depth": plan.depth,
            "sentence_difficulty": plan.sentence_difficulty,
            "frames": stats["frames"],
            "overrides": dict(plan.overrides),
        }

    def tts(self, text: str, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """One-shot convenience wrapper."""
        return self.synthesize(self.analyze(text), **kwargs)


def save_wav(path: str, wav: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.abs(wav).max())
    if peak > 1.0:
        wav = wav / peak
    sf.write(path, wav, sample_rate)
