"""Tests for collation, bucketing and the config loader.

Collation bugs are the classic silent training killer: shapes stay valid while
labels land on the wrong words. These tests pin the invariants.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from adaptts.data.dataset import LengthBucketSampler, Sample, collate  # noqa: E402
from adaptts.text.normalize import normalize_text, tokenize_words  # noqa: E402
from adaptts.text.vocab import CharVocab  # noqa: E402
from adaptts.utils.config import load_config, save_config  # noqa: E402


def _make_sample(text: str, vocab: CharVocab, codes_per_word, targets, T=6, Q=8, H=16):
    words, spans = tokenize_words(text)
    ids, widx = vocab.encode_with_word_index(text, spans)
    return Sample(
        uid="u" + text[:3],
        char_ids=np.asarray(ids, dtype=np.int64),
        word_index=np.asarray(widx, dtype=np.int64),
        n_codes=np.asarray(codes_per_word, dtype=np.int64),
        code_target=np.asarray(targets, dtype=np.int64),
        codes=np.random.randint(0, 32, (T, Q)).astype(np.int64),
        teacher_hidden=np.random.randn(len(words), H).astype(np.float32),
        n_words=len(words),
    )


def test_collate_puts_codes_on_the_right_characters():
    """The pronunciation code must land on exactly its own word's characters."""
    t1 = normalize_text("انا شوفت علم مصر")          # 4 words
    t2 = normalize_text("علم الفيزيا مهم")            # 3 words
    vocab = CharVocab.build([t1, t2], min_freq=1)

    # word 2 of t1 ("علم") is ambiguous with code 1; word 0 of t2 with code 0.
    s1 = _make_sample(t1, vocab, [1, 1, 3, 2], [-1, -1, 1, 0], T=6)
    s2 = _make_sample(t2, vocab, [3, 1, 1], [0, -1, -1], T=4)
    b = collate([s1, s2], vocab.pad_id, max_codes=4, n_quantizers=8, teacher_dim=16)

    assert b["char_ids"].shape == (2, max(len(s1.char_ids), len(s2.char_ids)))
    assert b["codes"].shape == (2, 6, 8)

    # Every character of t1 word 2 carries code 1; nothing else does.
    w1 = b["word_index"][0]
    got = b["pc_per_char"][0][w1 == 2]
    assert got.numel() > 0 and torch.all(got == 1), got
    assert torch.all(b["pc_per_char"][0][w1 == 0] == 4)  # unambiguous -> null slot
    assert torch.all(b["pc_per_char"][0][w1 < 0] == 4)   # spaces -> null slot

    # Padding is marked and frames past the true length are masked out.
    assert bool(b["char_padding_mask"][1, len(s2.char_ids) :].all())
    assert int(b["frame_mask"][1].sum()) == 4
    assert b["n_frames"].tolist() == [6, 4]


def test_collate_word_mask_and_targets_align():
    t1 = normalize_text("علم مصر دول")
    t2 = normalize_text("انا هنا")
    vocab = CharVocab.build([t1, t2], min_freq=1)
    s1 = _make_sample(t1, vocab, [3, 2, 2], [2, 1, 0])
    s2 = _make_sample(t2, vocab, [1, 1], [-1, -1])
    b = collate([s1, s2], vocab.pad_id, max_codes=4, n_quantizers=8, teacher_dim=16)

    assert b["word_mask"].tolist() == [[True, True, True], [True, True, False]]
    assert b["n_codes"][0].tolist() == [3, 2, 2]
    assert b["n_codes"][1].tolist() == [1, 1, 0]  # padded slot -> 0
    assert b["code_target"][0].tolist() == [2, 1, 0]
    assert b["code_target"][1].tolist() == [-1, -1, -1]
    assert b["teacher_hidden"].shape == (2, 3, 16)


def test_collate_leaves_unlabelled_ambiguous_words_unconditioned():
    """A word we could not label must not fake a code, it gets the null slot."""
    t = normalize_text("علم مصر")
    vocab = CharVocab.build([t], min_freq=1)
    s = _make_sample(t, vocab, [3, 2], [-1, -1])  # ambiguous but unlabelled
    b = collate([s], vocab.pad_id, max_codes=4, n_quantizers=8, teacher_dim=16)
    assert torch.all(b["pc_per_char"] == 4)


def test_length_bucket_sampler_covers_every_index_once():
    lengths = np.random.randint(10, 250, size=997)
    s = LengthBucketSampler(lengths, batch_size=16, boundaries=[32, 64, 96, 128, 160, 200, 240])
    batches = list(iter(s))
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(lengths))), "sampler dropped or duplicated indices"
    assert len(batches) == len(s)

    # Padding waste must be small, which is the entire point of bucketing.
    waste = sum(max(lengths[b]) * len(b) - sum(lengths[b]) for b in batches)
    total = sum(max(lengths[b]) * len(b) for b in batches)
    assert waste / total < 0.12, f"padding waste {waste / total:.1%} is too high"


def test_sampler_reshuffles_between_epochs():
    lengths = np.random.randint(10, 250, size=200)
    s = LengthBucketSampler(lengths, batch_size=8, boundaries=[64, 128], seed=1)
    s.set_epoch(0)
    a = [b[:] for b in iter(s)]
    s.set_epoch(1)
    c = [b[:] for b in iter(s)]
    assert a != c, "batches must differ across epochs"


def test_config_roundtrip_and_strictness():
    base = """
name: t
acoustic:
  n_layers: 12
  exit_layers: [4, 8, 12]
  exit_loss_weights: [0.3, 0.6, 1.0]
train:
  batch_size: 8
"""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.yaml")
        open(p, "w", encoding="utf-8").write(base)
        cfg = load_config(p)
        assert cfg.train.batch_size == 8
        assert cfg.acoustic.exit_layers == (4, 8, 12)

        # A typo must fail loudly at load time, not silently at hour three.
        open(p, "w", encoding="utf-8").write("train:\n  batch_sizee: 8\n")
        try:
            load_config(p)
        except ValueError as e:
            assert "unknown config key" in str(e)
        else:
            raise AssertionError("expected unknown-key rejection")

        # An inconsistent exit configuration must be rejected by validate().
        open(p, "w", encoding="utf-8").write(
            "acoustic:\n  n_layers: 12\n  exit_layers: [4, 8]\n"
            "  exit_loss_weights: [0.5, 1.0]\n"
        )
        try:
            load_config(p)
        except ValueError as e:
            assert "last exit layer" in str(e)
        else:
            raise AssertionError("expected exit-layer validation error")


def test_config_override_and_save():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.yaml")
        open(p, "w", encoding="utf-8").write("name: base\n")
        cfg = load_config(p, overrides={"train.batch_size": 64, "optim.lr": 1e-4})
        assert cfg.train.batch_size == 64 and abs(cfg.optim.lr - 1e-4) < 1e-12
        out = os.path.join(d, "saved.yaml")
        save_config(cfg, out)
        assert load_config(out).train.batch_size == 64


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
