"""Typed, hierarchical configuration.

Every experiment is one YAML file. Configs support ``_base_`` inheritance so an
experiment only states what differs from the base. Unknown keys raise, which
catches typos at load time instead of at hour three of training.
"""

from __future__ import annotations

import copy
import dataclasses
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, get_args, get_origin

import yaml

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PathsConfig:
    """All filesystem locations. Everything else is derived from ``root``."""

    root: str = "runs/exp1"
    dataset_dir: str = "data/masri100h"
    cache_dir: str = "cache/exp1"
    hf_dataset_id: str = "ehabnegm/100-hour-Egyptian-dataset-single-speaker"

    def resolve(self) -> "PathsConfig":
        out = copy.deepcopy(self)
        out.root = os.path.abspath(self.root)
        out.dataset_dir = os.path.abspath(self.dataset_dir)
        out.cache_dir = os.path.abspath(self.cache_dir)
        return out

    @property
    def ckpt_dir(self) -> Path:
        return Path(self.root) / "checkpoints"

    @property
    def tb_dir(self) -> Path:
        return Path(self.root) / "tensorboard"

    @property
    def sample_dir(self) -> Path:
        return Path(self.root) / "samples"

    @property
    def align_dir(self) -> Path:
        return Path(self.cache_dir) / "alignments"

    @property
    def codes_dir(self) -> Path:
        return Path(self.cache_dir) / "audio_codes"

    @property
    def teacher_dir(self) -> Path:
        return Path(self.cache_dir) / "teacher"

    @property
    def spanemb_dir(self) -> Path:
        return Path(self.cache_dir) / "span_embeddings"

    @property
    def lexicon_path(self) -> Path:
        """Discovered pronunciation-code table. Not handwritten."""
        return Path(self.cache_dir) / "pronunciation_codes.json"

    @property
    def charvocab_path(self) -> Path:
        return Path(self.cache_dir) / "char_vocab.json"

    @property
    def manifest_path(self) -> Path:
        return Path(self.cache_dir) / "manifest.jsonl"


@dataclass
class AudioConfig:
    sample_rate: int = 24000
    codec_sample_rate: int = 24000
    ssl_sample_rate: int = 16000
    frame_rate: float = 12.5  # Mimi
    n_quantizers: int = 8
    codebook_size: int = 2048
    min_duration: float = 1.0
    max_duration: float = 18.0


@dataclass
class TextConfig:
    max_chars: int = 320
    max_words: int = 72
    lowercase_latin: bool = True
    strip_diacritics: bool = True
    normalize_alef: bool = False  # keep orthographic distinctions by default
    normalize_digits: bool = True
    # Which column to read from a parquet dataset that ships several text
    # variants. Empty means: try the usual names in order.
    parquet_text_column: str = ""
    # Expand numbers, dates, URLs and abbreviations into spoken Egyptian words.
    use_egyptian_normalizer: bool = True


@dataclass
class TeacherConfig:
    """Frozen contextual LM used only during preprocessing."""

    model_id: str = "UBC-NLP/MARBERTv2"
    layers: Tuple[int, ...] = (8, 9, 10, 11)
    hidden_size: int = 768
    max_length: int = 256
    batch_size: int = 64
    dtype: str = "float16"


@dataclass
class AlignConfig:
    """CTC forced alignment settings."""

    model_id: str = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
    batch_size: int = 8
    chunk_seconds: float = 20.0
    min_word_seconds: float = 0.08
    max_word_seconds: float = 2.5
    score_threshold: float = -3.5  # mean log-prob per frame; below this, drop


@dataclass
class SpanEmbConfig:
    """Self-supervised acoustic embedding for word spans."""

    model_id: str = "facebook/wav2vec2-large-xlsr-53"
    layers: Tuple[int, ...] = (6, 7, 8, 9)
    batch_size: int = 8
    context_pad: float = 0.02  # seconds of padding around the span
    dim: int = 1024


@dataclass
class DiscoveryConfig:
    """Pronunciation-code discovery (clustering) hyperparameters."""

    min_word_freq: int = 12
    max_codes_per_word: int = 4
    n_bootstrap: int = 24
    bootstrap_frac: float = 0.8
    stability_threshold: float = 0.72  # Fowlkes-Mallows
    silhouette_threshold: float = 0.10
    min_separation: float = 0.55  # normalized centroid distance gate
    pca_dim: int = 48
    min_cluster_frac: float = 0.12  # a code must own >=12% of occurrences
    random_seed: int = 1234
    max_occurrences_per_word: int = 600


@dataclass
class ContextEncoderConfig:
    """The tiny student model that predicts pronunciation codes."""

    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 768
    dropout: float = 0.1
    max_codes: int = 4  # must be >= discovery.max_codes_per_word
    distill_weight: float = 1.0
    distill_temperature: float = 2.0
    ce_weight: float = 1.0
    difficulty_weight: float = 0.3
    label_smoothing: float = 0.02


@dataclass
class AcousticConfig:
    """RVQ code language model."""

    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    depth_d_model: int = 256
    depth_n_layers: int = 4
    depth_n_heads: int = 4
    text_d_model: int = 256
    text_n_layers: int = 4
    text_n_heads: int = 4
    speaker_dim: int = 192
    pc_embed_dim: int = 64
    exit_layers: Tuple[int, ...] = (4, 8, 12)
    layerdrop: float = 0.0
    self_distill_weight: float = 1.0
    exit_loss_weights: Tuple[float, ...] = (0.3, 0.6, 1.0)
    duration_weight: float = 0.5
    cfg_dropout: float = 0.1  # probability of dropping text cond during training
    monotonic_prior_weight: float = 1.0
    rvq_loss_weights: Tuple[float, ...] = (4.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


@dataclass
class OptimConfig:
    lr: float = 3e-4
    min_lr_ratio: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    max_steps: int = 120000
    accum_steps: int = 1
    fused: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 32
    eval_batch_size: int = 32
    num_workers: int = 6
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    precision: str = "bf16"  # bf16 | fp16 | fp32
    compile: bool = True
    compile_mode: str = "default"
    seed: int = 42
    log_every: int = 50
    eval_every: int = 2000
    save_every: int = 5000
    sample_every: int = 5000
    keep_last_n: int = 3
    bucket_boundaries: Tuple[int, ...] = (32, 64, 96, 128, 160, 200, 240)
    max_eval_batches: int = 50
    resume: str = ""


@dataclass
class InferenceConfig:
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    cfg_scale: float = 1.6
    tau_low: float = 0.15
    tau_high: float = 0.45
    force_depth: int = 0  # 0 = adaptive
    max_frames_ratio: float = 1.6
    min_frames_ratio: float = 0.6
    repetition_window: int = 12
    repetition_max_repeats: int = 3
    seed: int = 0


@dataclass
class Config:
    name: str = "exp1_homograph"
    paths: PathsConfig = field(default_factory=PathsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    text: TextConfig = field(default_factory=TextConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    spanemb: SpanEmbConfig = field(default_factory=SpanEmbConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    context_encoder: ContextEncoderConfig = field(default_factory=ContextEncoderConfig)
    acoustic: AcousticConfig = field(default_factory=AcousticConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    codec_model_id: str = "kyutai/mimi"

    def validate(self) -> None:
        a, c, ac = self.audio, self.context_encoder, self.acoustic
        if c.max_codes < self.discovery.max_codes_per_word:
            raise ValueError(
                f"context_encoder.max_codes ({c.max_codes}) must be >= "
                f"discovery.max_codes_per_word ({self.discovery.max_codes_per_word})"
            )
        if len(ac.rvq_loss_weights) != a.n_quantizers:
            raise ValueError(
                f"acoustic.rvq_loss_weights has {len(ac.rvq_loss_weights)} entries "
                f"but audio.n_quantizers is {a.n_quantizers}"
            )
        if len(ac.exit_layers) != len(ac.exit_loss_weights):
            raise ValueError("exit_layers and exit_loss_weights must have equal length")
        if any(e < 1 or e > ac.n_layers for e in ac.exit_layers):
            raise ValueError(f"exit_layers {ac.exit_layers} out of range 1..{ac.n_layers}")
        if list(ac.exit_layers) != sorted(ac.exit_layers):
            raise ValueError("exit_layers must be ascending")
        if ac.exit_layers[-1] != ac.n_layers:
            raise ValueError("the last exit layer must equal acoustic.n_layers")
        if ac.d_model % ac.n_heads != 0:
            raise ValueError("acoustic.d_model must be divisible by n_heads")
        if ac.depth_d_model % ac.depth_n_heads != 0:
            raise ValueError("acoustic.depth_d_model must be divisible by depth_n_heads")
        if ac.text_d_model % ac.text_n_heads != 0:
            raise ValueError("acoustic.text_d_model must be divisible by text_n_heads")
        if c.d_model % c.n_heads != 0:
            raise ValueError("context_encoder.d_model must be divisible by n_heads")
        if self.train.precision not in ("bf16", "fp16", "fp32"):
            raise ValueError(f"unknown precision {self.train.precision}")
        if not 0.0 <= self.inference.tau_low <= self.inference.tau_high <= 1.0:
            raise ValueError("require 0 <= tau_low <= tau_high <= 1")
        if self.discovery.min_cluster_frac <= 0 or self.discovery.min_cluster_frac >= 0.5:
            raise ValueError("discovery.min_cluster_frac must be in (0, 0.5)")


# ---------------------------------------------------------------------------
# YAML loading with _base_ inheritance and strict key checking
# ---------------------------------------------------------------------------


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml_with_base(path: Path, _seen: Optional[set] = None) -> Dict[str, Any]:
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular _base_ reference at {path}")
    _seen.add(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} must be a YAML mapping")
    base_ref = raw.pop("_base_", None)
    if base_ref is None:
        return raw
    if isinstance(base_ref, str):
        base_ref = [base_ref]
    merged: Dict[str, Any] = {}
    for ref in base_ref:
        ref_path = Path(ref)
        # An absolute _base_ is used as-is; a relative one resolves against the
        # directory of the file that referenced it.
        if not ref_path.is_absolute():
            ref_path = path.parent / ref_path
        merged = _deep_merge(merged, _load_yaml_with_base(ref_path, set(_seen)))
    return _deep_merge(merged, raw)


def _coerce(value: Any, target: Any, ctx: str) -> Any:
    """Coerce a YAML scalar/sequence to the annotated dataclass field type."""
    origin = get_origin(target)

    if target is Any:
        return value
    if origin is Union:
        args = [a for a in get_args(target) if a is not type(None)]
        if value is None:
            return None
        for a in args:
            try:
                return _coerce(value, a, ctx)
            except (TypeError, ValueError):
                continue
        raise TypeError(f"{ctx}: cannot coerce {value!r} to {target}")
    if origin in (tuple, Tuple):
        args = get_args(target)
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{ctx}: expected a sequence, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0], f"{ctx}[{i}]") for i, v in enumerate(value))
        if len(args) != len(value):
            raise ValueError(f"{ctx}: expected {len(args)} items, got {len(value)}")
        return tuple(_coerce(v, a, f"{ctx}[{i}]") for i, (v, a) in enumerate(zip(value, args)))
    if origin in (list, List):
        args = get_args(target)
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{ctx}: expected a sequence, got {type(value).__name__}")
        return [_coerce(v, args[0], f"{ctx}[{i}]") for i, v in enumerate(value)]
    if target is bool:
        if isinstance(value, bool):
            return value
        raise TypeError(f"{ctx}: expected bool, got {value!r}")
    if target is int:
        if isinstance(value, bool):
            raise TypeError(f"{ctx}: expected int, got bool")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and float(value).is_integer():
            return int(value)
        raise TypeError(f"{ctx}: expected int, got {value!r}")
    if target is float:
        if isinstance(value, bool):
            raise TypeError(f"{ctx}: expected float, got bool")
        if isinstance(value, (int, float)):
            return float(value)
        raise TypeError(f"{ctx}: expected float, got {value!r}")
    if target is str:
        if isinstance(value, str):
            return value
        raise TypeError(f"{ctx}: expected str, got {value!r}")
    if is_dataclass(target):
        if not isinstance(value, dict):
            raise TypeError(f"{ctx}: expected a mapping for {target.__name__}")
        return _from_dict(target, value, ctx)
    return value


def _from_dict(cls: Type[T], data: Dict[str, Any], ctx: str = "") -> T:
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        prefix = f"{ctx}." if ctx else ""
        raise ValueError(
            f"unknown config key(s) {sorted(prefix + u for u in unknown)}; "
            f"valid keys for {cls.__name__}: {sorted(known)}"
        )
    kwargs: Dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            continue
        sub_ctx = f"{ctx}.{name}" if ctx else name
        kwargs[name] = _coerce(data[name], f.type, sub_ctx)
    return cls(**kwargs)


def _resolve_annotations(cls: Type) -> None:
    """Resolve string annotations once so ``field.type`` is a real type."""
    import typing

    hints = typing.get_type_hints(cls)
    for f in dataclasses.fields(cls):
        if isinstance(f.type, str):
            f.type = hints[f.name]
        if is_dataclass(f.type):
            _resolve_annotations(f.type)


_resolve_annotations(Config)


def load_config(path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load a YAML config, apply dotted-key overrides, validate, and return it."""
    data = _load_yaml_with_base(Path(path))
    if overrides:
        for dotted, value in overrides.items():
            keys = dotted.split(".")
            node = data
            for k in keys[:-1]:
                node = node.setdefault(k, {})
                if not isinstance(node, dict):
                    raise ValueError(f"override {dotted}: {k} is not a mapping")
            node[keys[-1]] = value
    cfg = _from_dict(Config, data)
    cfg.paths = cfg.paths.resolve()
    cfg.validate()
    return cfg


def config_to_dict(cfg: Any) -> Dict[str, Any]:
    if is_dataclass(cfg):
        return {f.name: config_to_dict(getattr(cfg, f.name)) for f in fields(cfg)}
    if isinstance(cfg, (list, tuple)):
        return [config_to_dict(v) for v in cfg]
    return cfg


def save_config(cfg: Config, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False, allow_unicode=True)
