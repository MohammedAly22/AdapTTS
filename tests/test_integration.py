"""Integration test: the real scripts, on a synthetic dataset, offline.

Stages A1/A2/A4/A5 need large pretrained checkpoints (wav2vec2, MARBERTv2,
Mimi), so this test injects equivalent synthetic caches in exactly the formats
those stages emit, then runs the genuine discovery, dataset, collate and
training code on top. That exercises every path that a real run touches
downstream of the downloads, which is where integration bugs actually live.

    python tests/test_integration.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from adaptts.data.dataset import AdapTTSDataset, LengthBucketSampler, collate  # noqa: E402
from adaptts.data.discovery import PronunciationLexicon  # noqa: E402
from adaptts.data.preprocess import (  # noqa: E402
    OccurrenceRecord,
    build_char_vocab,
    build_manifest,
    discover_pronunciation_codes,
)
from adaptts.models.acoustic import AcousticModel, acoustic_loss  # noqa: E402
from adaptts.models.context_encoder import ContextEncoder, context_encoder_loss  # noqa: E402
from adaptts.text.normalize import tokenize_words  # noqa: E402
from adaptts.text.vocab import CharVocab  # noqa: E402
from adaptts.utils.config import load_config, save_config  # noqa: E402


def save_ragged(path: Path, arrays, dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lens = np.array([a.shape[0] for a in arrays], dtype=np.int64)
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    tail = arrays[0].shape[1:] if arrays and arrays[0].ndim > 1 else ()
    buf = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(int(offsets[-1]), *tail))
    for a, s in zip(arrays, offsets[:-1]):
        if a.shape[0]:
            buf[s : s + a.shape[0]] = a.astype(dtype, copy=False)
    buf.flush()
    np.save(path.with_name(path.stem + "_offsets.npy"), offsets)


def build_workspace(tmp: Path) -> Path:
    """Create the fake dataset and an experiment config pointing at it."""
    ds = tmp / "fake_ds"
    subprocess.run(
        [sys.executable, str(ROOT / "tests" / "make_fake_dataset.py"),
         "--out", str(ds), "--n", "48"],
        check=True, capture_output=True,
    )
    cfg_path = tmp / "it.yaml"
    cfg_path.write_text(
        f"_base_: {(ROOT / 'configs' / 'exp1_smoke.yaml').as_posix()}\n"
        f"name: integration\n"
        f"paths:\n"
        f"  root: {(tmp / 'run').as_posix()}\n"
        f"  dataset_dir: {ds.as_posix()}\n"
        f"  cache_dir: {(tmp / 'cache').as_posix()}\n",
        encoding="utf-8",
    )
    return cfg_path


def inject_caches(cfg, utts, vocab):
    """Write synthetic stand-ins for the model-dependent preprocessing stages.

    The formats here must match what stages A2, A4 and A5 really produce; if
    they drift, this test fails and tells us the contract broke.
    """
    rng = np.random.default_rng(0)

    # --- A2 stand-in: span embeddings for every word occurrence -----------
    # "علم" gets two acoustically distinct clusters keyed to its context, which
    # is exactly the structure real discovery must find.
    occ, vecs = [], []
    D = 64
    axis = rng.normal(size=D)
    axis /= np.linalg.norm(axis)
    row = 0
    for u in utts:
        words, _ = tokenize_words(u.text)
        for wi, w in enumerate(words):
            base = rng.normal(scale=0.25, size=D)
            if w == "علم":
                sci = "الفيزيا" in u.text or "الرياضيات" in u.text or "النفس" in u.text or "الاحياء" in u.text
                base = base + (6.0 if sci else -6.0) * axis
            occ.append(OccurrenceRecord(u.uid, wi, w, 0.0, 0.3, -0.5, row))
            vecs.append(base.astype(np.float32))
            row += 1
    embs = np.stack(vecs)

    # --- A5 stand-in: Mimi codes -----------------------------------------
    codes_dir = Path(cfg.paths.codes_dir)
    arrays, order = [], []
    for u in utts:
        T = max(6, int(u.duration * cfg.audio.frame_rate))
        arrays.append(rng.integers(0, cfg.audio.codebook_size,
                                   size=(T, cfg.audio.n_quantizers)).astype(np.int16))
        order.append(u.uid)
    save_ragged(codes_dir / "codes.npy", arrays, np.int16)
    (codes_dir / "uid_order.json").write_text(json.dumps(order), encoding="utf-8")

    # --- A4 stand-in: teacher word states ---------------------------------
    t_dir = Path(cfg.paths.teacher_dir)
    th, torder = [], []
    for u in utts:
        n = len(tokenize_words(u.text)[0])
        th.append(rng.normal(size=(n, cfg.teacher.hidden_size)).astype(np.float16))
        torder.append(u.uid)
    save_ragged(t_dir / "word_hidden.npy", th, np.float16)
    (t_dir / "uid_order.json").write_text(json.dumps(torder), encoding="utf-8")

    return occ, embs


def test_full_integration():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cfg_path = build_workspace(tmp)
        cfg = load_config(cfg_path)

        # --- real manifest + vocab, via the real code ---------------------
        utts = build_manifest(cfg)
        assert len(utts) == 48, f"expected 48 utterances, got {len(utts)}"
        vocab = build_char_vocab(cfg, utts)
        assert len(vocab) > 10

        occ, embs = inject_caches(cfg, utts, vocab)

        # --- real discovery ------------------------------------------------
        lexicon, labels = discover_pronunciation_codes(cfg, occ, embs)
        amb = lexicon.ambiguous_words
        assert "علم" in amb, (
            f"discovery failed to find the planted homograph; found {amb}"
        )
        assert lexicon.n_codes("علم") == 2, (
            f"expected 2 readings for the planted homograph, got {lexicon.n_codes('علم')}"
        )
        # It must not invent readings for genuinely unambiguous words.
        assert len(amb) == 1, f"discovery over-split: {amb}"

        # Labels must split cleanly along the planted context boundary.
        by_code = {}
        for (uid, wi), code in labels.items():
            text = next(u.text for u in utts if u.uid == uid)
            sci = any(k in text for k in ("الفيزيا", "الرياضيات", "النفس", "الاحياء"))
            by_code.setdefault(code, []).append(sci)
        for code, flags in by_code.items():
            purity = max(sum(flags), len(flags) - sum(flags)) / len(flags)
            assert purity > 0.95, f"code {code} mixes contexts (purity {purity:.2f})"

        (Path(cfg.paths.cache_dir) / "code_labels.json").write_text(
            json.dumps({f"{k[0]}\t{k[1]}": v for k, v in labels.items()}), encoding="utf-8"
        )

        # --- real dataset + collate ---------------------------------------
        train = AdapTTSDataset(cfg, "train", vocab, lexicon, need_codes=True, need_teacher=True)
        dev = AdapTTSDataset(cfg, "dev", vocab, lexicon, need_codes=True, need_teacher=True)
        assert len(train) > 0 and len(dev) > 0

        sampler = LengthBucketSampler(train.lengths, 4, cfg.train.bucket_boundaries)
        batch_idx = next(iter(sampler))
        batch = collate(
            [train[i] for i in batch_idx], vocab.pad_id,
            cfg.discovery.max_codes_per_word, cfg.audio.n_quantizers, cfg.teacher.hidden_size,
        )
        for key in ("char_ids", "codes", "pc_per_char", "n_codes", "teacher_hidden"):
            assert key in batch, f"collate dropped {key}"
        assert batch["codes"].shape[2] == cfg.audio.n_quantizers
        assert int(batch["codes"].max()) < cfg.audio.codebook_size
        # Some ambiguous word in this corpus must actually carry a code.
        assert int((batch["n_codes"] > 1).sum()) > 0

        # --- real training steps: context encoder --------------------------
        ctx = ContextEncoder(
            len(vocab), d_model=cfg.context_encoder.d_model,
            n_layers=cfg.context_encoder.n_layers, n_heads=cfg.context_encoder.n_heads,
            d_ff=cfg.context_encoder.d_ff, dropout=0.0,
            max_codes=cfg.context_encoder.max_codes,
            teacher_dim=cfg.teacher.hidden_size, pad_id=vocab.pad_id,
        )
        opt = torch.optim.AdamW(ctx.parameters(), lr=2e-3)
        first = last = None
        for _ in range(40):
            out = ctx(batch["char_ids"], batch["word_index"], batch["n_codes"],
                      batch["char_padding_mask"])
            loss, stats = context_encoder_loss(
                out, batch["code_target"], batch["word_mask"], batch["n_codes"],
                teacher_hidden=batch["teacher_hidden"], model=ctx,
                distill_weight=0.0, difficulty_weight=0.1, label_smoothing=0.0,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = first if first is not None else float(loss)
            last = float(loss)
        assert last < first, f"context encoder did not learn: {first:.3f} -> {last:.3f}"
        assert float(stats["code_acc"]) > 0.8, f"code acc {float(stats['code_acc'])}"

        # --- real training steps: acoustic model ---------------------------
        ac = AcousticModel(
            len(vocab), n_quantizers=cfg.audio.n_quantizers,
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
        aopt = torch.optim.AdamW(ac.parameters(), lr=1e-3)
        spk = torch.zeros(batch["char_ids"].shape[0], cfg.acoustic.speaker_dim)
        a_first = a_last = None
        for _ in range(8):
            out = ac(batch["char_ids"], batch["pc_per_char"], batch["codes"], spk,
                     batch["char_padding_mask"], batch["frame_mask"],
                     cfg_dropout=cfg.acoustic.cfg_dropout)
            loss, _ = acoustic_loss(
                out, batch["codes"], batch["frame_mask"], cfg.acoustic.rvq_loss_weights,
                cfg.acoustic.exit_loss_weights, batch["n_frames"].float(),
            )
            aopt.zero_grad()
            loss.backward()
            aopt.step()
            a_first = a_first if a_first is not None else float(loss)
            a_last = float(loss)
        assert a_last < a_first, f"acoustic model did not learn: {a_first:.3f} -> {a_last:.3f}"

        # --- real inference pipeline ---------------------------------------
        from adaptts.infer.pipeline import AdapTTS

        ac.eval()
        ctx.eval()
        tts = AdapTTS(cfg, vocab, lexicon, ctx, ac, codec=None, device=torch.device("cpu"))
        plan = tts.analyze("انا شوفت علم مصر بيرفرف")
        assert plan.hard_words, "the probe sentence should contain an ambiguous word"
        w = plan.hard_words[0]
        assert w.word == "علم" and w.n_codes == 2
        assert 0.0 <= w.confidence <= 1.0

        # Depth selection must respond to difficulty.
        assert tts.select_depth(0.0) == cfg.acoustic.exit_layers[0]
        assert tts.select_depth(1.0) == cfg.acoustic.exit_layers[-1]

        # Override must change the tensor handed to the acoustic model.
        before = plan.pc_per_char(cfg.discovery.max_codes_per_word).clone()
        plan.set_code("علم", 1 - w.code)
        after = plan.pc_per_char(cfg.discovery.max_codes_per_word)
        assert not torch.equal(before, after), "override did not reach the model input"

        # Generation must run at every trained depth and stay in range.
        with torch.no_grad():
            for depth in cfg.acoustic.exit_layers:
                codes, st = ac.generate(
                    plan.inputs["char_ids"], after, torch.zeros(1, cfg.acoustic.speaker_dim),
                    plan.inputs["char_padding_mask"], depth=depth,
                    max_frames=16, min_frames=4, cfg_scale=1.5,
                )
                assert int(codes.max()) < cfg.audio.codebook_size
                assert st["depth_used"] == float(depth)

        print("  discovery found:", lexicon.ambiguous_words)
        print(f"  context encoder loss {first:.3f} -> {last:.3f}, "
              f"code acc {float(stats['code_acc']):.3f}")
        print(f"  acoustic loss {a_first:.3f} -> {a_last:.3f}")

        # Release memory maps so the temp directory can be removed (Windows).
        train.close()
        dev.close()


if __name__ == "__main__":
    import io

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        test_full_integration()
        print("PASS  test_full_integration")
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"FAIL  test_full_integration: {exc}")
        sys.exit(1)
