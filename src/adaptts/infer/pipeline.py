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

from ..data.discovery import PronunciationLexicon
from ..models.acoustic import AcousticModel
from ..models.context_encoder import ContextEncoder
from ..text.normalize import normalize_text, tokenize_words
from ..text.vocab import CharVocab
from ..utils.config import Config, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------


def prepare_inputs(
    text: str,
    vocab: CharVocab,
    lexicon: Optional[PronunciationLexicon],
    device: torch.device,
    cfg: Optional[Config] = None,
) -> Dict[str, torch.Tensor]:
    """Normalize and tensorize one sentence for batch size 1."""
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
    return {
        "text": text,
        "words": words,
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

    # -- inspection ----------------------------------------------------

    @property
    def hard_words(self) -> List[WordPlan]:
        return [w for w in self.words if w.is_ambiguous]

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

    def set_depth(self, depth: int) -> "SynthesisPlan":
        self.depth = depth
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
        lexicon: Optional[PronunciationLexicon],
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
        lex_path = Path(cfg.paths.lexicon_path)
        lexicon = PronunciationLexicon.load(lex_path) if lex_path.exists() else None

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
            return SynthesisPlan(inp["text"], plans, 0.0,
                                 depth or self.cfg.acoustic.n_layers, inp)

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
        return SynthesisPlan(inp["text"], plans, sent_diff, chosen, inp)

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
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Render a plan (or a raw string) to a waveform."""
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
            temperature=temperature if temperature is not None else ic.temperature,
            top_k=ic.top_k, top_p=ic.top_p,
            cfg_scale=cfg_scale if cfg_scale is not None else ic.cfg_scale,
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
