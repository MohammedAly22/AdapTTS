"""Context encoder: the homograph disambiguator.

Input : a normalized character sequence plus word boundaries.
Output: for every word, a distribution over that word's pronunciation codes,
        plus a calibrated difficulty score.

This is the model that solves the central problem. It is deliberately tiny
(about 6M parameters) because it is distilled from MARBERTv2 rather than
competing with it, and because it must run in a few milliseconds on a CPU
before the acoustic model starts.

Two heads sit on top of the shared trunk:

* ``code_head``  - a *shared* projection to ``max_codes`` logits. Which of those
  logits are legal for a given word is determined by that word's discovered
  ``n_codes``; illegal ones are masked to -inf. A shared head means a rare
  homograph benefits from every other homograph's training signal, which is
  what makes this work on 68 hours of data.
* ``difficulty_head`` - predicts how uncertain the decision is. Trained against
  the entropy of the code posterior, so it is available before the codes are
  needed and can gate the acoustic model's depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.transformer import EncoderBlock, RMSNorm, RotaryEmbedding

# Large enough to zero the softmax in fp32, small enough that dividing by a
# distillation temperature or adding to another logit can never overflow.
NEG_INF = -1e4


@dataclass
class ContextEncoderOutput:
    code_logits: torch.Tensor  # [B, W, max_codes], illegal codes masked
    code_probs: torch.Tensor  # [B, W, max_codes]
    difficulty: torch.Tensor  # [B, W] in [0, 1]
    difficulty_logit: torch.Tensor  # [B, W] pre-sigmoid, for a stable BCE
    entropy: torch.Tensor  # [B, W] normalized posterior entropy
    word_hidden: torch.Tensor  # [B, W, d_model]
    char_hidden: torch.Tensor  # [B, L, d_model]


def pool_chars_to_words(
    char_hidden: torch.Tensor,
    word_index: torch.Tensor,
    n_words: int,
) -> torch.Tensor:
    """Mean-pool character states into word states.

    Args:
        char_hidden: ``[B, L, D]``
        word_index: ``[B, L]`` int64; the word each character belongs to, or -1.
        n_words: number of word slots ``W``.

    Returns:
        ``[B, W, D]``; words with no characters get zeros.
    """
    B, L, D = char_hidden.shape
    device = char_hidden.device
    valid = word_index >= 0
    idx = word_index.clamp(min=0)

    sums = torch.zeros(B, n_words, D, device=device, dtype=char_hidden.dtype)
    sums.scatter_add_(
        1, idx.unsqueeze(-1).expand(B, L, D), char_hidden * valid.unsqueeze(-1)
    )
    counts = torch.zeros(B, n_words, device=device, dtype=char_hidden.dtype)
    counts.scatter_add_(1, idx, valid.to(char_hidden.dtype))
    return sums / counts.clamp(min=1.0).unsqueeze(-1)


def broadcast_words_to_chars(
    word_feats: torch.Tensor,
    word_index: torch.Tensor,
) -> torch.Tensor:
    """Inverse of :func:`pool_chars_to_words`: scatter word features onto chars.

    Characters with ``word_index == -1`` (spaces, punctuation) receive zeros.
    """
    B, L = word_index.shape
    D = word_feats.shape[-1]
    valid = (word_index >= 0).unsqueeze(-1)
    idx = word_index.clamp(min=0).unsqueeze(-1).expand(B, L, D)
    return word_feats.gather(1, idx) * valid


class ContextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 768,
        dropout: float = 0.1,
        max_codes: int = 4,
        teacher_dim: int = 768,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_codes = max_codes
        self.pad_id = pad_id

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.rope = RotaryEmbedding(d_model // n_heads)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d_model)

        # Shared code head: rare homographs borrow statistical strength from
        # frequent ones instead of each needing its own classifier.
        self.code_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, max_codes)
        )
        self.difficulty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, 1)
        )
        # Projection used only in training, to align the student trunk with the
        # frozen teacher's contextual space.
        self.teacher_proj = nn.Linear(d_model, teacher_dim, bias=False)

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

    def trunk(
        self, char_ids: torch.Tensor, char_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embed(char_ids)
        cos, sin = self.rope(char_ids.shape[1], device=char_ids.device)
        for blk in self.blocks:
            x = blk(x, rope=(cos, sin), key_padding_mask=char_padding_mask)
        return self.norm(x)

    def forward(
        self,
        char_ids: torch.Tensor,
        word_index: torch.Tensor,
        n_codes_per_word: torch.Tensor,
        char_padding_mask: Optional[torch.Tensor] = None,
    ) -> ContextEncoderOutput:
        """
        Args:
            char_ids: ``[B, L]``
            word_index: ``[B, L]`` word id per character, -1 for non-word chars.
            n_codes_per_word: ``[B, W]`` how many codes each word legally has.
                A value of 1 means unambiguous. Zero marks a padding word slot.
            char_padding_mask: ``[B, L]`` True at padding.
        """
        if char_ids.dim() != 2:
            raise ValueError(f"char_ids must be [B, L], got {tuple(char_ids.shape)}")
        if word_index.shape != char_ids.shape:
            raise ValueError("word_index must have the same shape as char_ids")

        char_hidden = self.trunk(char_ids, char_padding_mask)
        W = n_codes_per_word.shape[1]
        word_hidden = pool_chars_to_words(char_hidden, word_index, W)

        logits = self.code_head(word_hidden)  # [B, W, max_codes]

        # Mask codes this word does not have. A word with n_codes = 1 becomes a
        # degenerate distribution with probability 1 on code 0 and zero entropy,
        # which is exactly the "this word is free" case.
        ar = torch.arange(self.max_codes, device=logits.device)
        legal = ar[None, None, :] < n_codes_per_word.clamp(min=1).unsqueeze(-1)
        logits = logits.masked_fill(~legal, NEG_INF)

        probs = logits.softmax(dim=-1)
        n_eff = n_codes_per_word.clamp(min=1).to(probs.dtype)
        ent = -(probs.clamp_min(1e-9).log() * probs).sum(-1)
        norm_ent = ent / torch.log(n_eff.clamp(min=2.0))
        norm_ent = torch.where(n_codes_per_word > 1, norm_ent, torch.zeros_like(norm_ent))

        # Keep the raw logit: binary_cross_entropy is unsafe under fp16 autocast,
        # so the loss uses the fused with-logits form instead of this sigmoid.
        diff_logit = self.difficulty_head(word_hidden).squeeze(-1)
        diff = torch.sigmoid(diff_logit)
        diff = torch.where(n_codes_per_word > 1, diff, torch.zeros_like(diff))

        return ContextEncoderOutput(
            code_logits=logits,
            code_probs=probs,
            difficulty=diff,
            difficulty_logit=diff_logit,
            entropy=norm_ent,
            word_hidden=word_hidden,
            char_hidden=char_hidden,
        )

    def sentence_difficulty(
        self, difficulty: torch.Tensor, word_mask: torch.Tensor, beta: float = 8.0
    ) -> torch.Tensor:
        """Soft-max pool per-word difficulty into one score per sentence.

        A plain mean would let a single hard word be diluted by thirty easy
        ones, which is the wrong behaviour: one hard word does require more
        compute. ``beta`` controls how sharply the pool approaches the max.
        """
        mask = word_mask.to(difficulty.dtype)
        w = torch.exp(beta * difficulty) * mask
        denom = w.sum(dim=1).clamp(min=1e-9)
        return (w * difficulty).sum(dim=1) / denom


def context_encoder_loss(
    out: ContextEncoderOutput,
    code_targets: torch.Tensor,
    word_mask: torch.Tensor,
    n_codes_per_word: torch.Tensor,
    teacher_logits: Optional[torch.Tensor] = None,
    teacher_hidden: Optional[torch.Tensor] = None,
    model: Optional[ContextEncoder] = None,
    ce_weight: float = 1.0,
    distill_weight: float = 1.0,
    distill_temperature: float = 2.0,
    difficulty_weight: float = 0.3,
    label_smoothing: float = 0.02,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Composite training objective for the context encoder.

    Only *ambiguous* words contribute to the code loss. Unambiguous words have a
    single legal code, so their cross-entropy is identically zero and including
    them would only dilute the gradient with easy examples.
    """
    device = out.code_logits.device
    ambiguous = (n_codes_per_word > 1) & word_mask & (code_targets >= 0)
    n_amb = ambiguous.sum()

    stats: Dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=device)

    if n_amb > 0:
        logits = out.code_logits[ambiguous]  # [N, max_codes], illegal = NEG_INF
        targets = code_targets[ambiguous]
        # log_softmax over the masked logits already assigns ~0 probability to
        # illegal codes, so NLL here is exactly CE restricted to legal codes.
        logp = F.log_softmax(logits.float(), dim=-1)
        if label_smoothing > 0:
            n_legal = (logits > NEG_INF / 2).sum(-1, keepdim=True).clamp(min=1).float()
            smooth = (logp * (logits > NEG_INF / 2)).sum(-1, keepdim=True) / n_legal
            ce = -(
                (1.0 - label_smoothing) * logp.gather(-1, targets.unsqueeze(-1))
                + label_smoothing * smooth
            ).mean()
        else:
            ce = F.nll_loss(logp, targets)
        total = total + ce_weight * ce
        stats["ce"] = ce.detach()
        with torch.no_grad():
            stats["code_acc"] = (logits.argmax(-1) == targets).float().mean()
    else:
        stats["ce"] = torch.zeros((), device=device)
        stats["code_acc"] = torch.zeros((), device=device)

    # Distillation from the frozen teacher's code posterior.
    if teacher_logits is not None and n_amb > 0:
        t = distill_temperature
        s_raw = out.code_logits[ambiguous].float()
        legal_m = s_raw > NEG_INF / 2
        # Divide before masking would scale NEG_INF too; mask after scaling so
        # illegal codes stay exactly out of the distribution at any temperature.
        s_log = F.log_softmax((s_raw / t).masked_fill(~legal_m, NEG_INF), dim=-1)
        t_log = F.log_softmax(
            (teacher_logits[ambiguous].float() / t).masked_fill(~legal_m, NEG_INF), dim=-1
        )
        kl = F.kl_div(s_log, t_log, reduction="batchmean", log_target=True) * (t * t)
        total = total + distill_weight * kl
        stats["kl"] = kl.detach()

    # Representation distillation: align the student's word states with the
    # teacher's contextual states. This is what transfers semantics rather than
    # only the final decision, and it is what lets the student generalize to
    # homographs that are rare in the audio.
    if teacher_hidden is not None and model is not None:
        m = word_mask.unsqueeze(-1)
        proj = model.teacher_proj(out.word_hidden)
        cos = F.cosine_similarity(
            (proj * m).float(), (teacher_hidden * m).float(), dim=-1
        )
        rep = (1.0 - cos)[word_mask].mean() if word_mask.any() else torch.zeros((), device=device)
        total = total + distill_weight * 0.5 * rep
        stats["rep"] = rep.detach()

    # Difficulty head regresses the (detached) normalized posterior entropy.
    #
    # Trained on AMBIGUOUS words only. On real text more than 95% of words have
    # a single reading and therefore a target of exactly 0, so averaging over
    # all of them lets the head reach a near-optimal loss by predicting 0
    # everywhere. Measured before this fix: a word with entropy 0.99 received a
    # predicted difficulty of 0.0018. Unambiguous words are already forced to 0
    # analytically in forward(), so excluding them here costs nothing.
    amb_mask = (n_codes_per_word > 1) & word_mask
    if difficulty_weight > 0 and bool(amb_mask.any()):
        target = out.entropy.detach()
        # with_logits rather than plain BCE: the plain form is rejected under
        # fp16 autocast and is worse conditioned even in fp32.
        d_loss = F.binary_cross_entropy_with_logits(
            out.difficulty_logit[amb_mask].float(),
            target[amb_mask].clamp(0.0, 1.0).float(),
        )
        total = total + difficulty_weight * d_loss
        stats["difficulty"] = d_loss.detach()
        with torch.no_grad():
            # Correlation between predicted difficulty and true entropy. If this
            # sits near zero the head has collapsed again.
            pred = out.difficulty[amb_mask].float()
            tgt = target[amb_mask].float()
            if pred.numel() > 1 and float(pred.std()) > 1e-6 and float(tgt.std()) > 1e-6:
                stats["difficulty_corr"] = (
                    ((pred - pred.mean()) * (tgt - tgt.mean())).mean()
                    / (pred.std() * tgt.std())
                )

    stats["loss"] = total.detach()
    stats["n_ambiguous"] = n_amb.detach().float()
    return total, stats
