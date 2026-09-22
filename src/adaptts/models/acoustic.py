"""Adaptive-depth RVQ acoustic model.

Predicts Mimi residual-vector-quantizer codes from text, pronunciation codes and
a speaker vector. Three parts:

1. **Text encoder** - bidirectional, consumes characters plus the per-word
   pronunciation-code embedding broadcast onto each character. This is where the
   homograph decision enters the acoustic path: the model does not see the
   ambiguous string, it sees the string *plus the resolved reading*.

2. **Temporal backbone** - causal, one step per 12.5 Hz frame, cross-attending
   to the text. Carries early-exit heads so inference can stop shallow on easy
   sentences.

3. **Depth transformer** - inside each frame, runs one micro-step per RVQ level,
   predicting level ``q`` given levels below it. Without this the levels are
   modelled independently, which is the usual source of codec artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.transformer import (
    DecoderBlock,
    EncoderBlock,
    KVCache,
    RMSNorm,
    RotaryEmbedding,
)


@dataclass
class AcousticOutput:
    logits: torch.Tensor  # [B, T, Q, codebook] from the deepest exit
    exit_logits: List[torch.Tensor]  # one tensor per early exit
    duration: torch.Tensor  # [B] predicted frame count
    align_weights: Optional[torch.Tensor]  # [B, T, S] text attention


def monotonic_alignment_bias(
    n_frames: int,
    n_text: int,
    device: torch.device,
    width: float = 0.12,
    strength: float = 1.0,
) -> torch.Tensor:
    """A soft diagonal prior over the frame-to-text attention.

    Speech is monotonic in the text. Biasing attention toward the diagonal is
    the cheapest robust defence against the two classic autoregressive TTS
    failures: skipping a span of text, and looping on one word forever. It is a
    *bias*, not a hard constraint, so the model can still deviate where the
    data demands it.
    """
    f = torch.arange(n_frames, device=device, dtype=torch.float32) / max(n_frames - 1, 1)
    s = torch.arange(n_text, device=device, dtype=torch.float32) / max(n_text - 1, 1)
    d = (f[:, None] - s[None, :]) ** 2
    return -strength * d / (2 * width * width)


class TextEncoder(nn.Module):
    """Characters + pronunciation codes -> a memory the decoder attends over."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        max_codes: int,
        pc_embed_dim: int,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # +1 slot: index `max_codes` means "this word is unambiguous / no code".
        self.pc_embed = nn.Embedding(max_codes + 1, pc_embed_dim)
        self.pc_null = max_codes
        self.pc_proj = nn.Linear(pc_embed_dim, d_model, bias=False)
        self.rope = RotaryEmbedding(d_model // n_heads)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        char_ids: torch.Tensor,
        pc_per_char: torch.Tensor,
        char_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            char_ids: ``[B, S]``
            pc_per_char: ``[B, S]`` pronunciation code of each character's word,
                or ``max_codes`` where the character belongs to no ambiguous word.
        """
        x = self.char_embed(char_ids) + self.pc_proj(self.pc_embed(pc_per_char))
        cos, sin = self.rope(char_ids.shape[1], device=char_ids.device)
        for blk in self.blocks:
            x = blk(x, rope=(cos, sin), key_padding_mask=char_padding_mask)
        return self.norm(x)


class DepthTransformer(nn.Module):
    """Predicts the Q RVQ levels of one frame, autoregressively over levels."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        n_quantizers: int,
        codebook_size: int,
        context_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_quantizers = n_quantizers
        self.codebook_size = codebook_size
        self.d_model = d_model

        self.ctx_proj = nn.Linear(context_dim, d_model, bias=False)
        # One embedding table per level keeps levels from sharing an index space.
        self.code_embed = nn.ModuleList(
            [nn.Embedding(codebook_size, d_model) for _ in range(n_quantizers)]
        )
        self.level_embed = nn.Embedding(n_quantizers, d_model)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, n_heads, d_ff, dropout, cross=False)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, codebook_size) for _ in range(n_quantizers)]
        )

    def forward(self, context: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Teacher-forced training pass.

        Args:
            context: ``[N, context_dim]`` backbone state for each frame.
            codes:   ``[N, Q]`` ground-truth codes for that frame.

        Returns:
            ``[N, Q, codebook_size]`` logits; position ``q`` is conditioned on
            the context and on levels ``< q`` only.
        """
        N, Q = codes.shape
        if Q != self.n_quantizers:
            raise ValueError(f"expected {self.n_quantizers} levels, got {Q}")

        steps = [self.ctx_proj(context)]
        for q in range(Q - 1):
            steps.append(self.code_embed[q](codes[:, q]))
        x = torch.stack(steps, dim=1)  # [N, Q, d]
        x = x + self.level_embed.weight[None, :Q, :]

        for blk in self.blocks:
            x, _ = blk(x)
        x = self.norm(x)
        return torch.stack([self.heads[q](x[:, q]) for q in range(Q)], dim=1)

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample all Q levels for one frame. ``context`` is ``[N, context_dim]``."""
        N = context.shape[0]
        device = context.device
        out = torch.zeros(N, self.n_quantizers, dtype=torch.long, device=device)
        steps = [self.ctx_proj(context)]
        for q in range(self.n_quantizers):
            x = torch.stack(steps, dim=1)
            x = x + self.level_embed.weight[None, : x.shape[1], :]
            for blk in self.blocks:
                x, _ = blk(x)
            h = self.norm(x)[:, -1]
            logits = self.heads[q](h)
            out[:, q] = sample_logits(logits, temperature, top_k, top_p, generator)
            if q < self.n_quantizers - 1:
                steps.append(self.code_embed[q](out[:, q]))
        return out


def sample_logits(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Nucleus + top-k sampling. ``logits`` is ``[N, V]``; returns ``[N]``."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits.float() / temperature

    if top_k and top_k < logits.shape[-1]:
        kth = logits.topk(top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        cum = probs.cumsum(dim=-1)
        remove = cum - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter_(1, sorted_idx, sorted_logits)

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


class AcousticModel(nn.Module):
    """Text + pronunciation codes + speaker -> Mimi RVQ codes."""

    def __init__(
        self,
        vocab_size: int,
        n_quantizers: int = 8,
        codebook_size: int = 2048,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        text_d_model: int = 256,
        text_n_layers: int = 4,
        text_n_heads: int = 4,
        depth_d_model: int = 256,
        depth_n_layers: int = 4,
        depth_n_heads: int = 4,
        speaker_dim: int = 192,
        max_codes: int = 4,
        pc_embed_dim: int = 64,
        exit_layers: Sequence[int] = (4, 8, 12),
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.n_quantizers = n_quantizers
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.exit_layers = tuple(exit_layers)
        self.max_codes = max_codes
        self.speaker_dim = speaker_dim

        self.text_encoder = TextEncoder(
            vocab_size, text_d_model, text_n_layers, text_n_heads,
            text_d_model * 3, max_codes, pc_embed_dim, dropout, pad_id,
        )
        # Unconditional text memory for classifier-free guidance. Learned, so
        # the guided direction is meaningful rather than an arbitrary zero.
        self.null_text = nn.Parameter(torch.zeros(1, 1, text_d_model))

        self.speaker_proj = nn.Linear(speaker_dim, d_model, bias=False)
        self.frame_in = nn.ModuleList(
            [nn.Embedding(codebook_size + 1, d_model) for _ in range(n_quantizers)]
        )
        self.bos_frame = nn.Parameter(torch.zeros(1, 1, d_model))

        self.rope = RotaryEmbedding(d_model // n_heads)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, n_heads, d_ff, dropout, cross=True, d_kv=text_d_model)
                for _ in range(n_layers)
            ]
        )
        # One norm per exit: a shallow exit needs its own output scaling.
        self.exit_norms = nn.ModuleList([RMSNorm(d_model) for _ in self.exit_layers])

        self.depth = DepthTransformer(
            depth_d_model, depth_n_layers, depth_n_heads, depth_d_model * 3,
            n_quantizers, codebook_size, d_model, dropout,
        )
        self.duration_head = nn.Sequential(
            nn.Linear(text_d_model, text_d_model // 2), nn.SiLU(),
            nn.Linear(text_d_model // 2, 1),
        )
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()

    # -- pieces -----------------------------------------------------------

    def encode_text(
        self,
        char_ids: torch.Tensor,
        pc_per_char: torch.Tensor,
        char_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.text_encoder(char_ids, pc_per_char, char_padding_mask)

    def predict_duration(
        self, mem: torch.Tensor, mem_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Expected frame count, used to bound generation length.

        The head predicts a speaking **rate** in frames per character, which is
        then multiplied by the real character count.

        Predicting the total directly does not work: the input is a mean-pooled
        text memory, and mean pooling divides by sequence length, so length is
        removed from the head's input by construction. Measured before this
        change, every sentence from 38 to 79 characters received the identical
        prediction of 26.2 frames against true lengths of 63 to 77.

        Splitting the problem fixes it. Length is arithmetic and needs no
        learning; rate is the part that actually depends on content, and it
        varies over a narrow, well-conditioned range.
        """
        if mem_padding_mask is not None:
            valid = (~mem_padding_mask).to(mem.dtype).unsqueeze(-1)
            n_tokens = valid.sum(1).squeeze(-1).clamp(min=1.0)
            pooled = (mem * valid).sum(1) / n_tokens.unsqueeze(-1)
        else:
            n_tokens = torch.full(
                (mem.shape[0],), float(mem.shape[1]), device=mem.device, dtype=mem.dtype
            )
            pooled = mem.mean(1)

        # Centred on a typical rate so the head starts near the right answer
        # and only has to learn the deviation. At 12.5 Hz, Egyptian speech runs
        # roughly one frame per character.
        rate = F.softplus(self.duration_head(pooled).squeeze(-1) + 0.55) + 0.15
        return (rate * n_tokens).clamp(min=4.0)

    def _frame_inputs(self, codes: torch.Tensor) -> torch.Tensor:
        """Embed a frame's Q codes into one backbone input vector."""
        h = self.frame_in[0](codes[..., 0])
        for q in range(1, self.n_quantizers):
            h = h + self.frame_in[q](codes[..., q])
        return h / self.n_quantizers

    # -- training ---------------------------------------------------------

    def forward(
        self,
        char_ids: torch.Tensor,
        pc_per_char: torch.Tensor,
        codes: torch.Tensor,
        speaker: torch.Tensor,
        char_padding_mask: Optional[torch.Tensor] = None,
        frame_padding_mask: Optional[torch.Tensor] = None,
        cfg_dropout: float = 0.0,
        monotonic_strength: float = 1.0,
        need_align: bool = False,
    ) -> AcousticOutput:
        """Teacher-forced forward pass.

        Args:
            codes: ``[B, T, Q]`` ground-truth Mimi codes.
            speaker: ``[B, speaker_dim]``.
        """
        B, T, Q = codes.shape
        if Q != self.n_quantizers:
            raise ValueError(f"expected {self.n_quantizers} quantizers, got {Q}")
        device = codes.device

        mem = self.encode_text(char_ids, pc_per_char, char_padding_mask)
        if cfg_dropout > 0 and self.training:
            drop = torch.rand(B, device=device) < cfg_dropout
            null = self.null_text.expand(B, mem.shape[1], -1)
            mem = torch.where(drop[:, None, None], null, mem)

        duration = self.predict_duration(mem, char_padding_mask)

        # Shift right: frame t is predicted from frames < t.
        prev = self._frame_inputs(codes)
        x = torch.cat([self.bos_frame.expand(B, 1, -1), prev[:, :-1]], dim=1)
        x = x + self.speaker_proj(speaker).unsqueeze(1)

        bias = monotonic_alignment_bias(
            T, mem.shape[1], device, strength=monotonic_strength
        )[None].expand(B, -1, -1) if monotonic_strength > 0 else None

        cos, sin = self.rope(T, device=device)
        exits: List[torch.Tensor] = []
        align = None
        exit_set = {l: i for i, l in enumerate(self.exit_layers)}

        for li, blk in enumerate(self.blocks, start=1):
            want = need_align and li == self.n_layers
            x, w = blk(
                x, rope=(cos, sin), mem=mem, mem_padding_mask=char_padding_mask,
                align_bias=bias, need_weights=want,
            )
            if want:
                align = w
            if li in exit_set:
                exits.append(self.exit_norms[exit_set[li]](x))

        # Depth transformer, run once per (batch, frame) pair.
        all_logits: List[torch.Tensor] = []
        flat_codes = codes.reshape(B * T, Q)
        for h in exits:
            logits = self.depth(h.reshape(B * T, -1), flat_codes)
            all_logits.append(logits.view(B, T, Q, self.codebook_size))

        return AcousticOutput(
            logits=all_logits[-1],
            exit_logits=all_logits,
            duration=duration,
            align_weights=align,
        )

    # -- generation -------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        char_ids: torch.Tensor,
        pc_per_char: torch.Tensor,
        speaker: torch.Tensor,
        char_padding_mask: Optional[torch.Tensor] = None,
        depth: Optional[int] = None,
        max_frames: Optional[int] = None,
        min_frames: Optional[int] = None,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        cfg_scale: float = 1.0,
        monotonic_strength: float = 1.0,
        repetition_window: int = 12,
        repetition_max_repeats: int = 3,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Autoregressively sample Mimi codes.

        Returns ``(codes [B, T, Q], stats)``.
        """
        B = char_ids.shape[0]
        device = char_ids.device
        depth = depth or self.n_layers
        if depth not in self.exit_layers:
            raise ValueError(f"depth {depth} is not one of exit_layers {self.exit_layers}")
        exit_idx = self.exit_layers.index(depth)

        mem = self.encode_text(char_ids, pc_per_char, char_padding_mask)
        duration = self.predict_duration(mem, char_padding_mask)
        # Duration-anchored bounds. The predicted frame count brackets how long
        # the utterance may run, which is what stops the two classic failures:
        # cutting off early, and looping forever.
        if max_frames is not None and int(max_frames) < 1:
            raise ValueError(f"max_frames must be at least 1, got {max_frames}")
        if min_frames is not None and int(min_frames) < 0:
            raise ValueError(f"min_frames cannot be negative, got {min_frames}")

        est = int(duration.max().item())
        hi = int(max_frames) if max_frames is not None else max(8, int(est * 1.6))
        lo = int(min_frames) if min_frames is not None else max(4, int(est * 0.6))
        # An explicit max_frames is a hard ceiling: clamp the floor to it rather
        # than growing the ceiling, so the caller's bound is never exceeded.
        lo = min(lo, hi)

        use_cfg = cfg_scale > 1.0
        if use_cfg:
            null = self.null_text.expand(B, mem.shape[1], -1)
            mem_cat = torch.cat([mem, null], dim=0)
            mask_cat = (
                torch.cat([char_padding_mask] * 2, dim=0)
                if char_padding_mask is not None else None
            )
            spk_cat = torch.cat([speaker, speaker], dim=0)
            eff_B = 2 * B
        else:
            mem_cat, mask_cat, spk_cat, eff_B = mem, char_padding_mask, speaker, B

        caches = [
            KVCache(eff_B, self.blocks[0].attn.n_heads, self.blocks[0].attn.head_dim,
                    hi + 2, device, mem.dtype)
            for _ in range(depth)
        ]
        spk = self.speaker_proj(spk_cat).unsqueeze(1)
        x = self.bos_frame.expand(eff_B, 1, -1) + spk

        out_codes: List[torch.Tensor] = []
        history: List[torch.Tensor] = []
        S = mem_cat.shape[1]
        stopped = torch.zeros(B, dtype=torch.bool, device=device)
        n_generated = 0

        # Precompute the full monotonic prior once; each step slices one row.
        full_bias = (
            monotonic_alignment_bias(hi, S, device, strength=monotonic_strength)
            if monotonic_strength > 0
            else None
        )

        for t in range(hi):
            # One query position per step, so the bias is [eff_B, 1, S].
            bias = (
                full_bias[t : t + 1].unsqueeze(0).expand(eff_B, 1, S)
                if full_bias is not None
                else None
            )

            cos, sin = self.rope(1, offset=t, device=device)
            h = x
            for li in range(depth):
                h, _ = self.blocks[li](
                    h, rope=(cos, sin), mem=mem_cat, mem_padding_mask=mask_cat,
                    align_bias=bias, cache=caches[li],
                )
            h = self.exit_norms[exit_idx](h)[:, 0]  # [eff_B, d]

            if use_cfg:
                # Guidance on the backbone state, before the depth transformer,
                # so both text-conditioned and unconditional paths share codes.
                cond, uncond = h[:B], h[B:]
                h_guided = uncond + cfg_scale * (cond - uncond)
                frame = self.depth.generate(h_guided, temperature, top_k, top_p, generator)
            else:
                frame = self.depth.generate(h, temperature, top_k, top_p, generator)

            # Repetition guard on the coarse (semantic) level.
            if repetition_window > 0 and len(history) >= repetition_window:
                recent = torch.stack(history[-repetition_window:], dim=1)  # [B, w]
                same = (recent == frame[:, :1]).sum(dim=1)
                loop = same >= repetition_max_repeats
                if loop.any():
                    # Resample the coarse level without its current choice.
                    ctx = h_guided if use_cfg else h
                    logits = self._coarse_logits(ctx)
                    logits.scatter_(1, frame[:, :1], float("-inf"))
                    frame[:, 0] = torch.where(
                        loop, sample_logits(logits, temperature, top_k, top_p, generator),
                        frame[:, 0],
                    )

            history.append(frame[:, 0].clone())
            out_codes.append(frame)
            n_generated = t + 1

            if t + 1 >= lo:
                # Silence-like frames repeating at the coarse level is the
                # model's natural end-of-speech signal.
                if len(history) >= 8:
                    tail = torch.stack(history[-8:], dim=1)
                    stopped = stopped | (tail == tail[:, :1]).all(dim=1)
                if bool(stopped.all()):
                    break

            # `frame` is [B, Q]; _frame_inputs gives [B, d]. Add the time axis
            # explicitly so each decode step feeds exactly one query position.
            nxt = self._frame_inputs(frame).unsqueeze(1) + spk[:B]
            x = torch.cat([nxt, nxt], dim=0) if use_cfg else nxt

        codes = torch.stack(out_codes, dim=1)
        return codes, {
            "frames": float(n_generated),
            "duration_estimate": float(duration.mean().item()),
            "depth_used": float(depth),
        }

    def _coarse_logits(self, context: torch.Tensor) -> torch.Tensor:
        """Level-0 logits for a frame context, used by the repetition guard."""
        x = self.depth.ctx_proj(context).unsqueeze(1)
        x = x + self.depth.level_embed.weight[None, :1, :]
        for blk in self.depth.blocks:
            x, _ = blk(x)
        return self.depth.heads[0](self.depth.norm(x)[:, 0])


def acoustic_loss(
    out: AcousticOutput,
    codes: torch.Tensor,
    frame_mask: torch.Tensor,
    rvq_weights: Sequence[float],
    exit_weights: Sequence[float],
    target_frames: Optional[torch.Tensor] = None,
    duration_weight: float = 0.5,
    self_distill_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Cross-entropy over RVQ levels at every exit, plus depth self-distillation.

    The self-distillation term is what makes early exits safe: the deepest exit
    is the teacher for every shallower one, so a shallow exit is a distilled
    model rather than a truncated one. Without it, exiting early produces
    garbage, which is exactly the hallucination failure mode we must avoid.
    """
    device = codes.device
    B, T, Q = codes.shape
    stats: Dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=device)

    w_rvq = torch.tensor(list(rvq_weights), device=device, dtype=torch.float32)
    w_rvq = w_rvq / w_rvq.sum()
    valid = frame_mask.reshape(-1)
    if not bool(valid.any()):
        raise ValueError("frame_mask selects no frames")

    deep_logits = out.exit_logits[-1]

    for ei, logits in enumerate(out.exit_logits):
        ce_total = torch.zeros((), device=device)
        for q in range(Q):
            lg = logits[:, :, q].reshape(-1, logits.shape[-1])[valid]
            tg = codes[:, :, q].reshape(-1)[valid]
            ce_total = ce_total + w_rvq[q] * F.cross_entropy(lg.float(), tg)
        ew = exit_weights[ei]
        total = total + ew * ce_total
        stats[f"ce_exit{ei}"] = ce_total.detach()

        if ei == len(out.exit_logits) - 1:
            with torch.no_grad():
                pred = logits[:, :, 0].reshape(-1, logits.shape[-1])[valid].argmax(-1)
                stats["acc_q0"] = (pred == codes[:, :, 0].reshape(-1)[valid]).float().mean()
        elif self_distill_weight > 0:
            kl_total = torch.zeros((), device=device)
            for q in range(Q):
                s = F.log_softmax(
                    logits[:, :, q].reshape(-1, logits.shape[-1])[valid].float(), -1
                )
                t = F.log_softmax(
                    deep_logits[:, :, q].reshape(-1, deep_logits.shape[-1])[valid].detach().float(), -1
                )
                kl_total = kl_total + w_rvq[q] * F.kl_div(
                    s, t, reduction="batchmean", log_target=True
                )
            total = total + self_distill_weight * ew * kl_total
            stats[f"kl_exit{ei}"] = kl_total.detach()

    if target_frames is not None and duration_weight > 0:
        d_loss = F.l1_loss(
            out.duration.float().log1p(), target_frames.float().log1p()
        )
        total = total + duration_weight * d_loss
        stats["duration"] = d_loss.detach()

    stats["loss"] = total.detach()
    return total, stats
