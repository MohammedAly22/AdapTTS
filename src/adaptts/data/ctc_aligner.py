"""CTC forced alignment, implemented from scratch.

Given a CTC acoustic model posterior ``log_probs[T, V]`` and a target label
sequence, we find the most likely monotonic frame-to-label assignment by Viterbi
search over the standard CTC expanded lattice (blank-separated labels).

Why from scratch rather than Montreal Forced Aligner: MFA requires a
pronunciation lexicon, and a pronunciation lexicon is exactly the handwritten
rule table this project forbids. A character CTC model needs no lexicon.

The implementation is vectorized over the label axis, so a 20 s utterance
aligns in a few milliseconds and the whole corpus is GPU-bound on the acoustic
model forward pass rather than on alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

NEG_INF = -1e30


@dataclass
class WordAlignment:
    """Frame-level span of one word, plus the alignment confidence."""

    word: str
    word_index: int
    start_frame: int
    end_frame: int  # exclusive
    score: float  # mean log-probability per frame

    def to_seconds(self, frames_per_second: float) -> Tuple[float, float]:
        return self.start_frame / frames_per_second, self.end_frame / frames_per_second


def _build_expanded_targets(targets: Sequence[int], blank: int) -> torch.Tensor:
    """Interleave blanks: ``[a, b]`` becomes ``[-, a, -, b, -]``."""
    out = torch.full((2 * len(targets) + 1,), blank, dtype=torch.long)
    if targets:
        out[1::2] = torch.tensor(list(targets), dtype=torch.long)
    return out


@torch.no_grad()
def ctc_forced_align(
    log_probs: torch.Tensor,
    targets: Sequence[int],
    blank: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Viterbi-align ``targets`` to ``log_probs``.

    Args:
        log_probs: ``[T, V]`` log-softmax acoustic posteriors.
        targets: label ids (no blanks); must be non-empty.
        blank: blank label id.

    Returns:
        ``(path, scores)`` where ``path[t]`` is the index into the *expanded*
        target sequence assigned to frame ``t``, and ``scores[t]`` is the
        log-probability emitted at that frame.
    """
    if log_probs.dim() != 2:
        raise ValueError(f"log_probs must be [T, V], got {tuple(log_probs.shape)}")
    if len(targets) == 0:
        raise ValueError("targets must be non-empty")

    device = log_probs.device
    log_probs = log_probs.float()
    T, V = log_probs.shape
    ext = _build_expanded_targets(targets, blank).to(device)
    S = ext.numel()

    if T < S - S // 2:  # need at least one frame per non-blank label
        raise ValueError(
            f"utterance too short to align: {T} frames for {len(targets)} labels"
        )

    # emission[t, s] = log P(ext[s] | frame t)
    emission = log_probs.index_select(1, ext)  # [T, S]

    # A transition s-2 -> s is allowed only when ext[s] is a non-blank label
    # different from ext[s-2]; that is the CTC "skip blank" rule.
    can_skip = torch.zeros(S, dtype=torch.bool, device=device)
    if S > 2:
        non_blank = ext != blank
        differs = ext[2:] != ext[:-2]
        can_skip[2:] = non_blank[2:] & differs

    alpha = torch.full((S,), NEG_INF, device=device)
    alpha[0] = emission[0, 0]
    if S > 1:
        alpha[1] = emission[0, 1]

    backptr = torch.zeros((T, S), dtype=torch.int8, device=device)

    for t in range(1, T):
        stay = alpha
        prev = torch.cat([torch.full((1,), NEG_INF, device=device), alpha[:-1]])
        skip = torch.cat([torch.full((2,), NEG_INF, device=device), alpha[:-2]])
        skip = torch.where(can_skip, skip, torch.full_like(skip, NEG_INF))

        cand = torch.stack([stay, prev, skip], dim=0)  # [3, S]
        best, arg = cand.max(dim=0)
        alpha = best + emission[t]
        backptr[t] = arg.to(torch.int8)

    # The path must end on the last label or the final blank.
    if S > 1:
        end_state = int(S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2)
    else:
        end_state = 0

    path = torch.zeros(T, dtype=torch.long, device=device)
    s = end_state
    for t in range(T - 1, -1, -1):
        path[t] = s
        move = int(backptr[t, s].item())
        if t > 0:
            s = s - move  # 0 stay, 1 prev, 2 skip
            if s < 0:
                raise RuntimeError("CTC backtrace left the lattice; this is a bug")

    scores = emission.gather(1, path.unsqueeze(1)).squeeze(1)
    return path, scores


def spans_from_path(
    path: torch.Tensor,
    scores: torch.Tensor,
    n_targets: int,
) -> List[Tuple[int, int, float]]:
    """Convert an expanded-lattice path into per-label frame spans.

    Expanded index ``2*i + 1`` corresponds to target label ``i``. Blank frames
    belong to no label. Returns ``(start, end, mean_score)`` per label; a label
    that received no frame gets ``(-1, -1, -inf)``.
    """
    out: List[Tuple[int, int, float]] = []
    path_cpu = path.detach().cpu()
    scores_cpu = scores.detach().cpu()
    label_states = path_cpu.numpy()
    for i in range(n_targets):
        state = 2 * i + 1
        mask = label_states == state
        idx = mask.nonzero()[0]
        if idx.size == 0:
            out.append((-1, -1, float("-inf")))
            continue
        start, end = int(idx[0]), int(idx[-1]) + 1
        mean_score = float(scores_cpu[start:end].mean().item())
        out.append((start, end, mean_score))
    return out


def word_spans_from_char_spans(
    char_spans: Sequence[Tuple[int, int, float]],
    word_of_char: Sequence[int],
    words: Sequence[str],
) -> List[WordAlignment]:
    """Aggregate character spans into word spans.

    ``word_of_char[i]`` gives the word index of aligned character ``i``, or -1.
    A word's span runs from the first frame of its first aligned character to
    the last frame of its last aligned character. Characters that received no
    frames are skipped; a word with no aligned characters is omitted entirely.
    """
    if len(char_spans) != len(word_of_char):
        raise ValueError(
            f"char_spans ({len(char_spans)}) and word_of_char ({len(word_of_char)}) "
            "must have equal length"
        )

    acc: dict = {}
    for ci, (start, end, score) in enumerate(char_spans):
        wi = word_of_char[ci]
        if wi < 0 or start < 0:
            continue
        if wi not in acc:
            acc[wi] = [start, end, [score]]
        else:
            entry = acc[wi]
            entry[0] = min(entry[0], start)
            entry[1] = max(entry[1], end)
            entry[2].append(score)

    out: List[WordAlignment] = []
    for wi in sorted(acc):
        start, end, scs = acc[wi]
        if wi >= len(words):
            raise ValueError(f"word index {wi} out of range for {len(words)} words")
        out.append(
            WordAlignment(
                word=words[wi],
                word_index=wi,
                start_frame=start,
                end_frame=end,
                score=float(sum(scs) / len(scs)),
            )
        )
    return out


def enforce_monotonic(alignments: Sequence[WordAlignment]) -> List[WordAlignment]:
    """Drop any word whose span violates monotonicity (defensive; CTC is monotonic)."""
    out: List[WordAlignment] = []
    last_end = -1
    for a in alignments:
        if a.start_frame < last_end:
            continue
        out.append(a)
        last_end = a.end_frame
    return out
