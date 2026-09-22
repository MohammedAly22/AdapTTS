"""Shared training utilities: optimizer, schedule, checkpoints, AMP, logging."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..utils.config import Config, config_to_dict

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_backend() -> None:
    """Enable the fast matmul paths. Worth roughly 20-30% on Ada GPUs."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW with weight decay applied only to matmul weights.

    Norm gains, biases and embeddings are excluded, which is standard practice
    and measurably better than decaying everything on small models.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias") or "norm" in name.lower() or "embed" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    groups = [
        {"params": decay, "weight_decay": cfg.optim.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs: Dict[str, Any] = dict(lr=cfg.optim.lr, betas=cfg.optim.betas, eps=1e-8)
    if cfg.optim.fused and torch.cuda.is_available():
        try:
            return torch.optim.AdamW(groups, fused=True, **kwargs)
        except (RuntimeError, TypeError):
            logger.warning("fused AdamW unavailable; using the standard implementation")
    return torch.optim.AdamW(groups, **kwargs)


def lr_at_step(step: int, cfg: Config) -> float:
    """Linear warmup then cosine decay to ``min_lr_ratio * lr``."""
    warm, total = cfg.optim.warmup_steps, cfg.optim.max_steps
    base, floor = cfg.optim.lr, cfg.optim.lr * cfg.optim.min_lr_ratio
    if step < warm:
        return base * (step + 1) / max(warm, 1)
    if step >= total:
        return floor
    prog = (step - warm) / max(total - warm, 1)
    return floor + 0.5 * (base - floor) * (1.0 + math.cos(math.pi * prog))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def amp_context(precision: str, device: torch.device):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def maybe_compile(model: nn.Module, cfg: Config) -> nn.Module:
    if not cfg.train.compile:
        return model
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile unavailable in this build; skipping")
        return model
    try:
        compiled = torch.compile(model, mode=cfg.train.compile_mode)
        logger.info("torch.compile enabled (mode=%s)", cfg.train.compile_mode)
        return compiled
    except Exception as exc:  # noqa: BLE001
        logger.warning("torch.compile failed (%s); continuing uncompiled", exc)
        return model


class CheckpointManager:
    """Saves checkpoints, keeps the last N, and tracks the best metric."""

    def __init__(self, directory: Path, keep_last_n: int = 3, higher_is_better: bool = False) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep = keep_last_n
        self.higher_is_better = higher_is_better
        self.best: Optional[float] = None
        self.saved: List[Path] = sorted(self.dir.glob("step_*.pt"))

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        cfg: Config,
        metrics: Dict[str, float],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        raw = getattr(model, "_orig_mod", model)  # unwrap torch.compile
        payload = {
            "model": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config_to_dict(cfg),
            "metrics": metrics,
        }
        if extra:
            payload.update(extra)

        path = self.dir / f"step_{step:08d}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        self.saved.append(path)

        while len(self.saved) > self.keep:
            old = self.saved.pop(0)
            old.unlink(missing_ok=True)

        key = metrics.get("monitor")
        if key is not None:
            better = (
                self.best is None
                or (key > self.best if self.higher_is_better else key < self.best)
            )
            if better:
                self.best = key
                shutil.copyfile(path, self.dir / "best.pt")
                logger.info("new best checkpoint at step %d (monitor=%.4f)", step, key)

        shutil.copyfile(path, self.dir / "last.pt")
        return path

    @staticmethod
    def load(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> int:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        raw = getattr(model, "_orig_mod", model)
        raw.load_state_dict(ckpt["model"])
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        logger.info("resumed from %s at step %d", path, ckpt.get("step", 0))
        return int(ckpt.get("step", 0))


class MetricTracker:
    """Running means over the logging interval."""

    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, Any]) -> None:
        for k, v in metrics.items():
            val = float(v.item()) if isinstance(v, torch.Tensor) else float(v)
            if not math.isfinite(val):
                continue
            self.sums[k] = self.sums.get(k, 0.0) + val
            self.counts[k] = self.counts.get(k, 0) + 1

    def mean(self) -> Dict[str, float]:
        return {k: self.sums[k] / max(self.counts[k], 1) for k in self.sums}

    def reset(self) -> None:
        self.sums.clear()
        self.counts.clear()


class Throughput:
    """Steps and items per second, for spotting a starved GPU."""

    def __init__(self) -> None:
        self.t0 = time.time()
        self.steps = 0
        self.items = 0

    def update(self, items: int) -> None:
        self.steps += 1
        self.items += items

    def rate(self) -> Tuple[float, float]:
        dt = max(time.time() - self.t0, 1e-9)
        return self.steps / dt, self.items / dt

    def reset(self) -> None:
        self.t0 = time.time()
        self.steps = 0
        self.items = 0


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def gpu_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**2)
