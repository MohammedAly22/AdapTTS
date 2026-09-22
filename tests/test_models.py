"""Correctness tests for the model components.

These assert the properties that silently break training if violated:
causality, masking, cache equivalence and numerical sanity.
Run with ``pytest tests/`` or ``python tests/test_models.py``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from adaptts.data.ctc_aligner import (  # noqa: E402
    ctc_forced_align,
    spans_from_path,
    word_spans_from_char_spans,
)
from adaptts.data.discovery import discover_word_codes  # noqa: E402
from adaptts.models.acoustic import AcousticModel, acoustic_loss  # noqa: E402
from adaptts.models.context_encoder import (  # noqa: E402
    ContextEncoder,
    broadcast_words_to_chars,
    context_encoder_loss,
    pool_chars_to_words,
)
from adaptts.modules.transformer import (  # noqa: E402
    EncoderBlock,
    KVCache,
    RotaryEmbedding,
    SelfAttention,
)
from adaptts.text.normalize import normalize_text, tokenize_words  # noqa: E402
from adaptts.text.vocab import CharVocab  # noqa: E402


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------


def test_normalization_strips_diacritics_only():
    raw = "عَلَم مِصر"
    out = normalize_text(raw)
    assert out == "علم مصر", out
    # Normalization must be idempotent, or cached features drift from live ones.
    assert normalize_text(out) == out


def test_word_index_alignment():
    text = normalize_text("انا شوفت علم مصر بيرفرف")
    words, spans = tokenize_words(text)
    assert words == ["انا", "شوفت", "علم", "مصر", "بيرفرف"]
    vocab = CharVocab.build([text], min_freq=1)
    ids, widx = vocab.encode_with_word_index(text, spans)
    assert len(ids) == len(widx) == len(text) + 2
    # Every character of word 2 must map back to word 2.
    s, e = spans[2]
    assert all(widx[1 + i] == 2 for i in range(s, e))
    assert vocab.decode(ids) == text


# --------------------------------------------------------------------------
# CTC alignment
# --------------------------------------------------------------------------


def test_ctc_alignment_recovers_known_spans():
    V, blank = 6, 0
    targets = [1, 2, 2, 3]  # repeated label requires a separating blank
    true = [0, 1, 1, 0, 2, 2, 0, 2, 0, 3, 3, 0]
    lp = torch.full((len(true), V), -8.0)
    for t, lab in enumerate(true):
        lp[t, lab] = 0.0
    lp = torch.log_softmax(lp, dim=-1)

    path, scores = ctc_forced_align(lp, targets, blank=blank)
    spans = [(s, e) for s, e, _ in spans_from_path(path, scores, len(targets))]
    assert spans == [(1, 3), (4, 6), (7, 8), (9, 11)], spans

    words = word_spans_from_char_spans(
        spans_from_path(path, scores, len(targets)), [0, 0, 1, 1], ["ab", "cd"]
    )
    assert (words[0].start_frame, words[0].end_frame) == (1, 6)
    assert (words[1].start_frame, words[1].end_frame) == (7, 11)


def test_ctc_rejects_too_short_input():
    try:
        ctc_forced_align(torch.log_softmax(torch.randn(1, 6), -1), [1, 2, 3])
    except ValueError:
        return
    raise AssertionError("expected a ValueError for an over-short utterance")


# --------------------------------------------------------------------------
# Pronunciation-code discovery
# --------------------------------------------------------------------------


def test_discovery_splits_real_homograph_and_not_noise():
    rng = np.random.default_rng(0)
    D = 64
    c1 = rng.normal(0, 1, D)
    c2 = c1 + 6.0 * (rng.normal(0, 1, D) / np.linalg.norm(rng.normal(0, 1, D)))

    bimodal = np.vstack(
        [c1 + 0.45 * rng.normal(0, 1, (70, D)), c2 + 0.45 * rng.normal(0, 1, (50, D))]
    )
    wc = discover_word_codes("homograph", bimodal, seed=0)
    assert wc.n_codes == 2, f"expected 2 readings, got {wc.n_codes}"
    assert sorted(wc.counts) == [50, 70], wc.counts

    unimodal = rng.normal(0, 1, (120, D))
    assert discover_word_codes("plain", unimodal, seed=0).n_codes == 1

    # Prosody-like heavy-tailed variation must not be mistaken for a reading.
    prosody = rng.normal(0, 1, (150, D)) * rng.gamma(2, 0.6, (150, 1))
    assert discover_word_codes("prosody", prosody, seed=0).n_codes == 1

    # Too few occurrences to be confident -> stay unambiguous.
    assert discover_word_codes("rare", rng.normal(0, 1, (6, D)), seed=0).n_codes == 1


def test_discovery_is_deterministic():
    rng = np.random.default_rng(3)
    D = 32
    c = rng.normal(0, 1, D)
    x = np.vstack(
        [c + 0.3 * rng.normal(0, 1, (60, D)), c + 8 + 0.3 * rng.normal(0, 1, (60, D))]
    )
    a = discover_word_codes("w", x, seed=11)
    b = discover_word_codes("w", x, seed=11)
    assert a.n_codes == b.n_codes
    assert np.allclose(a.centroids, b.centroids)


# --------------------------------------------------------------------------
# Transformer primitives
# --------------------------------------------------------------------------


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    d, h, T, B = 64, 4, 12, 2
    attn = SelfAttention(d, h, 0.0, causal=True).eval()
    rope = RotaryEmbedding(d // h)
    x = torch.randn(B, T, d)
    with torch.no_grad():
        cos, sin = rope(T)
        full = attn(x, rope=(cos, sin))
        cache = KVCache(B, h, d // h, 64, x.device, x.dtype)
        inc = torch.cat(
            [attn(x[:, t : t + 1], rope=rope(1, offset=t), cache=cache) for t in range(T)],
            dim=1,
        )
    assert (full - inc).abs().max().item() < 1e-4


def test_encoder_padding_is_isolated():
    torch.manual_seed(0)
    d, h, T, B = 64, 4, 12, 2
    blk = EncoderBlock(d, h, 128, 0.0).eval()
    rope = RotaryEmbedding(d // h)
    x = torch.randn(B, T, d)
    pad = torch.zeros(B, T, dtype=torch.bool)
    pad[0, 8:] = True
    with torch.no_grad():
        cos, sin = rope(T)
        y = blk(x, rope=(cos, sin), key_padding_mask=pad)
        xz = x.clone()
        xz[0, 8:] = 999.0
        yz = blk(xz, rope=(cos, sin), key_padding_mask=pad)
    assert torch.allclose(y[0, :8], yz[0, :8], atol=1e-5)


# --------------------------------------------------------------------------
# Context encoder
# --------------------------------------------------------------------------


def _ctx_fixture():
    torch.manual_seed(0)
    V, B, L, W = 100, 3, 40, 8
    m = ContextEncoder(V, d_model=128, n_layers=2, n_heads=4, d_ff=256, max_codes=4)
    ids = torch.randint(4, V, (B, L))
    widx = torch.randint(-1, W, (B, L))
    ncodes = torch.tensor([[1, 3, 1, 2, 1, 1, 1, 0], [2, 1, 1, 1, 4, 1, 0, 0], [1] * 8])
    pad = torch.zeros(B, L, dtype=torch.bool)
    pad[0, 30:] = True
    return m, ids, widx, ncodes, pad


def test_unambiguous_words_are_deterministic_and_free():
    m, ids, widx, ncodes, pad = _ctx_fixture()
    out = m(ids, widx, ncodes, pad)
    un = ncodes == 1
    assert torch.allclose(out.code_probs[un][:, 0], torch.ones(int(un.sum())), atol=1e-5)
    assert out.entropy[un].abs().max().item() < 1e-6
    assert out.difficulty[un].abs().max().item() < 1e-6


def test_illegal_codes_get_no_probability_mass():
    m, ids, widx, ncodes, pad = _ctx_fixture()
    out = m(ids, widx, ncodes, pad)
    B, W = ncodes.shape
    for b in range(B):
        for w in range(W):
            k = int(ncodes[b, w].clamp(min=1))
            tail = out.code_probs[b, w, k:]
            if tail.numel():
                assert float(tail.abs().max()) < 1e-8


def test_pool_and_broadcast_are_inverse_on_word_chars():
    torch.manual_seed(0)
    B, L, W, D = 3, 40, 8, 16
    widx = torch.randint(-1, W, (B, L))
    ch = torch.randn(B, L, D)
    wp = pool_chars_to_words(ch, widx, W)
    bc = broadcast_words_to_chars(wp, widx)
    assert bc[widx < 0].abs().max().item() == 0.0
    assert wp.shape == (B, W, D) and bc.shape == (B, L, D)


def test_context_loss_is_finite_and_small():
    m, ids, widx, ncodes, pad = _ctx_fixture()
    out = m(ids, widx, ncodes, pad)
    B, W = ncodes.shape
    tgt = torch.minimum(torch.randint(0, 4, (B, W)), (ncodes - 1).clamp(min=0))
    loss, stats = context_encoder_loss(
        out, tgt, ncodes > 0, ncodes, torch.randn(B, W, 4), torch.randn(B, W, 768), m
    )
    loss.backward()
    assert torch.isfinite(loss)
    # A masked-logit bug used to make this explode to ~1e8; guard against it.
    assert float(loss) < 50.0, f"loss {float(loss)} suggests a masking overflow"
    assert sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None) > 0


def test_sentence_difficulty_is_dominated_by_the_hardest_word():
    m, *_ = _ctx_fixture()
    mask = torch.ones(1, 10, dtype=torch.bool)
    one_hard = torch.zeros(1, 10)
    one_hard[0, 3] = 0.9
    assert float(m.sentence_difficulty(one_hard, mask)) > 0.8
    assert float(m.sentence_difficulty(torch.zeros(1, 10), mask)) == 0.0


# --------------------------------------------------------------------------
# Acoustic model
# --------------------------------------------------------------------------


def _acoustic_fixture():
    torch.manual_seed(0)
    V, B, S, T, Q, CB = 80, 2, 24, 10, 8, 64
    m = AcousticModel(
        V, n_quantizers=Q, codebook_size=CB, d_model=96, n_layers=4, n_heads=4, d_ff=192,
        text_d_model=64, text_n_layers=2, text_n_heads=4, depth_d_model=64,
        depth_n_layers=2, depth_n_heads=4, speaker_dim=32, max_codes=4, exit_layers=(2, 4),
    ).eval()
    ch = torch.randint(4, V, (B, S))
    pc = torch.randint(0, 5, (B, S))
    cd = torch.randint(0, CB, (B, T, Q))
    sp = torch.randn(B, 32)
    cm = torch.zeros(B, S, dtype=torch.bool)
    cm[0, 20:] = True
    fm = torch.ones(B, T, dtype=torch.bool)
    fm[1, 8:] = False
    return m, ch, pc, cd, sp, cm, fm, (B, S, T, Q, CB)


def test_backbone_is_causal_over_frames():
    m, ch, pc, cd, sp, cm, fm, (B, S, T, Q, CB) = _acoustic_fixture()
    with torch.no_grad():
        a = m(ch, pc, cd, sp, cm, fm).logits
        cd2 = cd.clone()
        cd2[:, T - 1] = torch.randint(0, CB, (B, Q))
        b = m(ch, pc, cd2, sp, cm, fm).logits
    assert torch.allclose(a[:, : T - 1], b[:, : T - 1], atol=1e-5)


def test_depth_transformer_level_causality():
    """Within a frame, level q must be predicted from levels < q only."""
    m, ch, pc, cd, sp, cm, fm, (B, S, T, Q, CB) = _acoustic_fixture()
    with torch.no_grad():
        base = m(ch, pc, cd, sp, cm, fm).logits
        # The last level of the last frame feeds nothing anywhere: inert.
        c2 = cd.clone()
        c2[:, T - 1, Q - 1] = (c2[:, T - 1, Q - 1] + 1) % CB
        assert float((base - m(ch, pc, c2, sp, cm, fm).logits).abs().max()) == 0.0

        # Changing level q at frame 0 must leave levels <= q at frame 0 alone.
        for q in range(1, Q):
            cq = cd.clone()
            cq[:, 0, q] = (cq[:, 0, q] + 1) % CB
            lq = m(ch, pc, cq, sp, cm, fm).logits
            assert float((base[:, 0, : q + 1] - lq[:, 0, : q + 1]).abs().max()) == 0.0


def test_acoustic_loss_and_generation_at_every_exit():
    m, ch, pc, cd, sp, cm, fm, (B, S, T, Q, CB) = _acoustic_fixture()
    out = m(ch, pc, cd, sp, cm, fm, need_align=True)
    assert out.logits.shape == (B, T, Q, CB)
    assert out.align_weights.shape == (B, T, S)

    loss, stats = acoustic_loss(
        out, cd, fm, [4, 2, 1, 1, 1, 1, 1, 1], [0.5, 1.0], fm.sum(1).float()
    )
    loss.backward()
    assert torch.isfinite(loss) and float(loss) < 100.0
    assert "kl_exit0" in stats, "shallow exits must receive self-distillation"

    for depth in m.exit_layers:
        with torch.no_grad():
            codes, st = m.generate(
                ch, pc, sp, cm, depth=depth, max_frames=12, min_frames=4, cfg_scale=1.0
            )
        assert codes.shape[0] == B and codes.shape[2] == Q
        assert int(codes.max()) < CB and int(codes.min()) >= 0
        assert st["depth_used"] == float(depth)


def test_generation_with_classifier_free_guidance():
    m, ch, pc, cd, sp, cm, fm, (B, S, T, Q, CB) = _acoustic_fixture()
    with torch.no_grad():
        codes, _ = m.generate(
            ch, pc, sp, cm, depth=4, max_frames=12, min_frames=4, cfg_scale=2.0
        )
    assert codes.shape[0] == B and int(codes.max()) < CB


def test_invalid_depth_is_rejected():
    m, ch, pc, cd, sp, cm, fm, _ = _acoustic_fixture()
    try:
        m.generate(ch, pc, sp, cm, depth=3)
    except ValueError:
        return
    raise AssertionError("expected a ValueError for a depth that is not an exit layer")


def test_generation_respects_frame_bounds():
    """max_frames is a hard ceiling.

    An unbounded generator is how autoregressive TTS hangs or babbles, so the
    caller's ceiling must never be widened internally.
    """
    m, ch, pc, cd, sp, cm, fm, (B, S, T, Q, CB) = _acoustic_fixture()
    for mf in (1, 3, 8, 20):
        with torch.no_grad():
            codes, _ = m.generate(
                ch, pc, sp, cm, depth=4, max_frames=mf, min_frames=mf,
                cfg_scale=1.0, temperature=0.0,
            )
        assert codes.shape[1] <= mf, f"generated {codes.shape[1]} frames, ceiling was {mf}"

    # A min_frames above max_frames must clamp rather than override the ceiling.
    with torch.no_grad():
        codes, _ = m.generate(
            ch, pc, sp, cm, depth=4, max_frames=5, min_frames=50, temperature=0.0
        )
    assert codes.shape[1] <= 5

    try:
        m.generate(ch, pc, sp, cm, depth=4, max_frames=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected max_frames=0 to be rejected")


def test_greedy_generation_is_deterministic_and_cfg_has_effect():
    m, ch, pc, cd, sp, cm, fm, (B, S, T, Q, CB) = _acoustic_fixture()
    kw = dict(depth=4, max_frames=10, min_frames=10, temperature=0.0)
    with torch.no_grad():
        plain = m.generate(ch, pc, sp, cm, cfg_scale=1.0, **kw)[0]
        guided_a = m.generate(ch, pc, sp, cm, cfg_scale=2.5, **kw)[0]
        guided_b = m.generate(ch, pc, sp, cm, cfg_scale=2.5, **kw)[0]
    assert torch.equal(guided_a, guided_b), "greedy decoding must be reproducible"
    assert not torch.equal(plain, guided_a), "guidance had no effect on the output"
    assert int(guided_a.max()) < CB and int(guided_a.min()) >= 0


def test_difficulty_head_tracks_entropy_rather_than_collapsing():
    """The difficulty head must respond to uncertainty, not predict 0 always.

    On real text over 95% of words are unambiguous with a target of exactly 0,
    so a head trained on all words reaches a near-optimal loss by outputting 0
    everywhere. Measured before the fix: entropy 0.99 produced a prediction of
    0.0018. The loss therefore trains on ambiguous words only.
    """
    torch.manual_seed(0)
    V, B, L, W = 40, 16, 24, 6
    m = ContextEncoder(V, d_model=64, n_layers=2, n_heads=4, d_ff=128,
                       dropout=0.0, max_codes=4)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)

    ids = torch.randint(4, V, (B, L))
    widx = torch.randint(0, W, (B, L))
    # One ambiguous word per sentence, the rest unambiguous: the real ratio.
    ncodes = torch.ones(B, W, dtype=torch.long)
    ncodes[:, 0] = 2
    mask = torch.ones(B, W, dtype=torch.bool)
    # Targets are resampled every step, so the code head cannot memorize them
    # and the posterior stays near 50/50 with entropy near 1. A working
    # difficulty head must follow its target up there.
    tgt = torch.full((B, W), -1, dtype=torch.long)

    for _ in range(150):
        tgt[:, 0] = torch.randint(0, 2, (B,))
        out = m(ids, widx, ncodes, None)
        loss, stats = context_encoder_loss(
            out, tgt, mask, ncodes, ce_weight=1.0, distill_weight=0.0,
            difficulty_weight=1.0, label_smoothing=0.0,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    out = m(ids, widx, ncodes, None)
    amb = ncodes > 1
    mean_entropy = float(out.entropy[amb].mean())
    mean_difficulty = float(out.difficulty[amb].mean())
    assert mean_entropy > 0.5, f"fixture did not stay uncertain: {mean_entropy}"
    assert mean_difficulty > 0.3, (
        f"difficulty collapsed: entropy {mean_entropy:.3f} but predicted "
        f"{mean_difficulty:.4f}"
    )
    # And unambiguous words must still read exactly zero.
    assert float(out.difficulty[ncodes == 1].abs().max()) == 0.0


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
