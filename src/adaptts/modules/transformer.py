"""Transformer building blocks tuned for small models and CPU inference.

Design choices that matter for the CPU target:

* **RMSNorm** instead of LayerNorm: one fewer reduction, no bias.
* **SwiGLU** feed-forward: better quality per parameter than GELU MLP.
* **Rotary position embeddings**: no learned position table, so the model
  extrapolates to longer utterances than it saw in training.
* **F.scaled_dot_product_attention**: dispatches to flash/mem-efficient kernels
  on GPU and to an optimized fused path on CPU.
* **Incremental KV cache** with pre-allocated buffers, so autoregressive decoding
  does no per-step allocation.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings with a cached cos/sin table."""

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"rotary head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.base = base
        self._build(max_seq_len)

    def _build(self, seq_len: int) -> None:
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)
        self._max_seq_len = seq_len

    def forward(self, seq_len: int, offset: int = 0, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        need = seq_len + offset
        if need > self._max_seq_len:
            self._build(int(2 ** math.ceil(math.log2(max(need, 16)))))
        cos = self.cos_cached[offset : offset + seq_len]
        sin = self.sin_cached[offset : offset + seq_len]
        if device is not None and cos.device != device:
            cos, sin = cos.to(device), sin.to(device)
        return cos, sin


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to ``x`` of shape ``[B, H, T, D]``."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos.to(x.dtype)[None, None, :, :]
    sin = sin.to(x.dtype)[None, None, :, :]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class KVCache:
    """Pre-allocated key/value cache for incremental decoding."""

    def __init__(
        self,
        batch: int,
        n_heads: int,
        head_dim: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.k = torch.zeros(batch, n_heads, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(batch, n_heads, max_len, head_dim, device=device, dtype=dtype)
        self.length = 0
        self.max_len = max_len

    def append(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n = k.shape[2]
        if self.length + n > self.max_len:
            raise RuntimeError(
                f"KV cache overflow: {self.length + n} > {self.max_len}. "
                "Increase max_len when constructing the cache."
            )
        self.k[:, :, self.length : self.length + n] = k
        self.v[:, :, self.length : self.length + n] = v
        self.length += n
        return self.k[:, :, : self.length], self.v[:, :, : self.length]

    def reset(self) -> None:
        self.length = 0


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, causal: bool = True) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if rope is not None:
            cos, sin = rope
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

        if cache is not None:
            k, v = cache.append(k, v)

        attn_mask = None
        is_causal = self.causal and cache is None and T > 1
        if key_padding_mask is not None:
            # key_padding_mask: [B, S] with True at padded keys
            S = k.shape[2]
            if key_padding_mask.shape[1] != S:
                raise ValueError(
                    f"key_padding_mask has {key_padding_mask.shape[1]} keys but attention has {S}"
                )
            bias = torch.zeros(B, 1, T, S, device=x.device, dtype=q.dtype)
            bias = bias.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
            if self.causal and cache is None and T > 1:
                causal = torch.ones(T, S, device=x.device, dtype=torch.bool).tril(diagonal=S - T)
                bias = bias.masked_fill(~causal[None, None], float("-inf"))
                is_causal = False
            attn_mask = bias

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


class CrossAttention(nn.Module):
    """Cross-attention with an optional additive alignment bias.

    The bias slot is what carries the monotonic alignment prior in the acoustic
    model; keeping it additive means it costs one add and never changes shapes.
    """

    def __init__(self, d_model: int, n_heads: int, d_kv: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        d_kv = d_kv or d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_kv, 2 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        mem_padding_mask: Optional[torch.Tensor] = None,
        align_bias: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.shape
        S = mem.shape[1]
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(mem).chunk(2, dim=-1)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        bias = None
        if mem_padding_mask is not None or align_bias is not None:
            bias = torch.zeros(B, 1, T, S, device=x.device, dtype=q.dtype)
            if align_bias is not None:
                if align_bias.shape != (B, T, S):
                    raise ValueError(
                        f"align_bias must be [B, T, S] = {(B, T, S)}, got {tuple(align_bias.shape)}"
                    )
                bias = bias + align_bias[:, None].to(q.dtype)
            if mem_padding_mask is not None:
                bias = bias.masked_fill(mem_padding_mask[:, None, None, :], float("-inf"))

        if need_weights:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = (q @ k.transpose(-2, -1)) * scale
            if bias is not None:
                scores = scores + bias
            w = scores.softmax(dim=-1)
            y = w @ v
            weights = w.mean(dim=1)  # [B, T, S] head-averaged
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0
            )
            weights = None

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y), weights


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class EncoderBlock(nn.Module):
    """Pre-norm bidirectional block (used by the text/context encoders)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.n1 = RMSNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout, causal=False)
        self.n2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.drop(self.attn(self.n1(x), rope=rope, key_padding_mask=key_padding_mask))
        x = x + self.ff(self.n2(x))
        return x


class DecoderBlock(nn.Module):
    """Pre-norm causal block with optional cross-attention to the text memory."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        cross: bool = True,
        d_kv: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.n1 = RMSNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout, causal=True)
        self.has_cross = cross
        if cross:
            self.n_cross = RMSNorm(d_model)
            self.cross = CrossAttention(d_model, n_heads, d_kv=d_kv, dropout=dropout)
        self.n2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mem: Optional[torch.Tensor] = None,
        mem_padding_mask: Optional[torch.Tensor] = None,
        align_bias: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = x + self.drop(self.attn(self.n1(x), rope=rope, cache=cache))
        weights = None
        if self.has_cross:
            if mem is None:
                raise ValueError("cross-attention block requires `mem`")
            h, weights = self.cross(
                self.n_cross(x), mem, mem_padding_mask, align_bias, need_weights=need_weights
            )
            x = x + self.drop(h)
        x = x + self.ff(self.n2(x))
        return x, weights
