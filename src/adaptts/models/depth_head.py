"""Parallel RVQ level predictor.

## Why this replaces the depth transformer

The original design ran a small transformer once per quantizer level inside
every frame, so each frame cost ``n_layers * n_quantizers`` block calls: 32 with
the shipped configuration. Profiled on CPU that was 21 ms per frame and 72 to
80% of total inference time, against 6.3 ms for the entire 12-layer backbone.

The cost was not arithmetic. One block on a single ``d=256`` token needs about
67 microseconds of matmul and took 453 microseconds through PyTorch modules, a
7x dispatch overhead, and a batch of 8 cost barely more than a batch of 1. That
is the signature of per-operator overhead, not compute, and no kernel tuning
fixes it. The only real remedy is to make fewer calls.

## What replaces it, and what that costs

All ``Q`` levels are predicted from the backbone state in one shot:

    backbone state  ->  shared trunk (2 layers)  ->  Q output heads

That is 2 module calls per frame instead of 32.

The honest trade-off: the per-level transformer let level ``q`` attend to the
sampled values of levels below it, which keeps the residual levels mutually
consistent. Predicting them independently can produce combinations the codec
never saw.

Two things preserve most of that consistency at negligible cost:

* **Coarse-to-fine conditioning.** Level ``q``'s head receives the trunk state
  plus the *embedding of the level below it*. During generation the levels are
  still produced in order, so level 1 sees the sampled level 0. This keeps the
  chain that matters while adding one embedding lookup per level rather than
  four transformer blocks.
* **Level embeddings.** Each head is conditioned on which level it predicts, so
  a single trunk can specialize without separate trunks.

Levels beyond the first few carry fine acoustic detail where cross-level
consistency matters much less than it does for the coarse semantic level, which
is why the reduced coupling is acceptable.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.transformer import RMSNorm


class ParallelDepthHead(nn.Module):
    """Predict all RVQ levels of a frame with a shared trunk and per-level heads.

    Args:
        context_dim: width of the backbone state.
        d_model: width of the shared trunk.
        n_quantizers: number of RVQ levels.
        codebook_size: entries per level.
        n_layers: trunk depth. Two is enough; the trunk only has to reshape one
            vector, and every extra layer is another dispatch per frame.
        cond_dim: width of the coarse-to-fine conditioning embedding.
    """

    def __init__(
        self,
        context_dim: int,
        d_model: int,
        n_quantizers: int,
        codebook_size: int,
        n_layers: int = 2,
        cond_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_quantizers = n_quantizers
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.cond_dim = cond_dim

        layers: List[nn.Module] = [nn.Linear(context_dim, d_model, bias=False)]
        for _ in range(max(0, n_layers - 1)):
            layers += [RMSNorm(d_model), nn.SiLU(), nn.Linear(d_model, d_model, bias=False)]
        self.trunk = nn.Sequential(*layers)
        self.trunk_norm = RMSNorm(d_model)

        # Which level a head predicts.
        self.level_embed = nn.Embedding(n_quantizers, d_model)

        # Coarse-to-fine conditioning: level q sees the code chosen at q-1.
        # One embedding lookup per level replaces four transformer blocks.
        self.prev_embed = nn.ModuleList(
            [nn.Embedding(codebook_size, cond_dim) for _ in range(n_quantizers - 1)]
        )
        self.prev_proj = nn.Linear(cond_dim, d_model, bias=False)

        self.heads = nn.ModuleList(
            [nn.Linear(d_model, codebook_size) for _ in range(n_quantizers)]
        )
        self.dropout = nn.Dropout(dropout)

    def _level_state(
        self, trunk: torch.Tensor, q: int, prev_code: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Trunk state specialized for level ``q``."""
        h = trunk + self.level_embed.weight[q][None, :]
        if q > 0 and prev_code is not None:
            h = h + self.prev_proj(self.prev_embed[q - 1](prev_code))
        return self.dropout(F.silu(h))

    def forward(self, context: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Teacher-forced training pass.

        Args:
            context: ``[N, context_dim]`` backbone state per frame.
            codes:   ``[N, Q]`` ground-truth codes for that frame.

        Returns:
            ``[N, Q, codebook_size]``. Level ``q`` sees the ground-truth code at
            ``q-1`` and nothing above it, so the coarse-to-fine ordering holds.
        """
        N, Q = codes.shape
        if Q != self.n_quantizers:
            raise ValueError(f"expected {self.n_quantizers} levels, got {Q}")

        trunk = self.trunk_norm(self.trunk(context))
        outs = []
        for q in range(Q):
            prev = codes[:, q - 1] if q > 0 else None
            outs.append(self.heads[q](self._level_state(trunk, q, prev)))
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        generator: Optional[torch.Generator] = None,
        caches: Optional[list] = None,  # accepted for interface compatibility
    ) -> torch.Tensor:
        """Sample all Q levels for one frame.

        The trunk runs once; each level then costs one embedding lookup, a few
        elementwise ops and one head matmul. That is the whole point: the
        expensive part is amortized over all levels instead of repeated.
        """
        from .acoustic import sample_logits

        N = context.shape[0]
        out = torch.zeros(N, self.n_quantizers, dtype=torch.long, device=context.device)
        trunk = self.trunk_norm(self.trunk(context))

        prev: Optional[torch.Tensor] = None
        for q in range(self.n_quantizers):
            logits = self.heads[q](self._level_state(trunk, q, prev))
            out[:, q] = sample_logits(logits, temperature, top_k, top_p, generator)
            prev = out[:, q]
        return out

    def make_caches(self, batch: int, device, dtype) -> list:
        """No caches needed. Present so callers need not special-case this."""
        return []
