"""Pre-rental readiness check: verify every known failure mode is fixed.

Run this immediately after cloning on a rented GPU, before spending money:

    python scripts/preflight_check.py --config configs/exp1_egyptian.yaml

It checks the things that actually stop a run: model files that cannot load,
the CTC blank index, autocast safety, picklable dataloaders, the duration head,
the difficulty head, and GPU precision support. Each check states what it
verified, so a pass means something specific rather than "no exception".
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from adaptts.utils.console import enable_utf8_console  # noqa: E402

enable_utf8_console()

RESULTS = []


def check(name: str):
    """Decorator that runs one check and records pass/fail with detail."""

    def wrap(fn):
        try:
            detail = fn()
            RESULTS.append((True, name, detail or ""))
            print(f"  PASS  {name}")
            if detail:
                for line in str(detail).splitlines():
                    print(f"        {line}")
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((False, name, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL  {name}")
            print(f"        {type(exc).__name__}: {exc}")
            if os.environ.get("PREFLIGHT_TRACE"):
                traceback.print_exc()
        return fn

    return wrap


def main() -> None:
    ap = argparse.ArgumentParser(description="AdapTTS preflight check")
    ap.add_argument("--config", default="configs/exp1_egyptian.yaml")
    ap.add_argument("--skip-downloads", action="store_true",
                    help="skip the checks that fetch pretrained models")
    args = ap.parse_args()

    from adaptts.utils.config import load_config

    cfg = load_config(args.config)

    print()
    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)

    @check("PyTorch version is new enough")
    def _torch_version():
        parts = torch.__version__.split("+")[0].split(".")
        major, minor = int(parts[0]), int(parts[1])
        if (major, minor) < (2, 4):
            raise RuntimeError(
                f"torch {torch.__version__} is too old. This code uses the "
                "torch.amp.GradScaler signature introduced in 2.4. Upgrade with:\n"
                "  pip install --upgrade torch torchaudio "
                "--index-url https://download.pytorch.org/whl/cu121"
            )
        note = ""
        if (major, minor) < (2, 6):
            note = (
                "\nnote: transformers 5.x refuses .bin checkpoints below torch 2.6 "
                "(CVE-2025-32434).\nThe configured models all ship safetensors, so "
                "this is fine, but a model swap may not be."
            )
        return f"torch {torch.__version__}{note}"

    @check("PyTorch sees a GPU")
    def _gpu():
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device visible")
        p = torch.cuda.get_device_properties(0)
        cc = p.major + p.minor / 10
        rec = "bf16" if cc >= 8.0 else "fp16"
        return (
            f"{p.name}, {p.total_memory / 1024 ** 3:.1f} GB, compute {p.major}.{p.minor}\n"
            f"recommended train.precision: {rec} (config says {cfg.train.precision})"
        )

    @check("configured precision suits this GPU")
    def _prec():
        if not torch.cuda.is_available():
            return "no GPU, skipped"
        p = torch.cuda.get_device_properties(0)
        cc = p.major + p.minor / 10
        # torch reports is_bf16_supported() True on Turing, but that is
        # emulation and is slower than fp16 there.
        if cfg.train.precision == "bf16" and cc < 8.0:
            raise RuntimeError(
                f"compute {cc} has no hardware bf16; set train.precision to fp16"
            )
        if cfg.train.precision == "fp16" and cc >= 8.0:
            return "fp16 works, but bf16 is faster and needs no gradient scaler here"
        return f"{cfg.train.precision} is correct for compute {cc}"

    print()
    print("=" * 70)
    print("KNOWN FAILURE MODES")
    print("=" * 70)

    @check("dataloader collate is picklable (Windows spawn)")
    def _pickle():
        import pickle

        from adaptts.data.dataset import Collator

        c = Collator(0, 4, 8, 768)
        pickle.loads(pickle.dumps(c))
        return "Collator survives pickling, so num_workers > 0 can start"

    @check("difficulty loss is autocast-safe and does not collapse")
    def _difficulty():
        from adaptts.models.context_encoder import ContextEncoder, context_encoder_loss

        m = ContextEncoder(40, d_model=64, n_layers=2, n_heads=4, d_ff=128,
                           dropout=0.0, max_codes=4)
        ids = torch.randint(4, 40, (4, 16))
        widx = torch.randint(0, 4, (4, 16))
        nc = torch.ones(4, 4, dtype=torch.long)
        nc[:, 0] = 2
        out = m(ids, widx, nc, None)
        if not hasattr(out, "difficulty_logit"):
            raise RuntimeError("difficulty_logit missing: the fp16 BCE fix is absent")
        tgt = torch.full((4, 4), -1, dtype=torch.long)
        tgt[:, 0] = 1
        loss, stats = context_encoder_loss(
            out, tgt, nc > 0, nc, difficulty_weight=1.0
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss is not finite: {loss}")
        if float(loss) > 100:
            raise RuntimeError(f"loss {float(loss):.1f} suggests a masking overflow")
        return f"uses binary_cross_entropy_with_logits, loss {float(loss):.3f}"

    @check("duration head responds to text length")
    def _duration():
        from adaptts.models.acoustic import AcousticModel

        m = AcousticModel(40, n_quantizers=8, codebook_size=64, d_model=64,
                          n_layers=2, n_heads=4, d_ff=128, text_d_model=32,
                          text_n_layers=1, text_n_heads=4, depth_d_model=32,
                          depth_n_layers=1, depth_n_heads=4, speaker_dim=16,
                          max_codes=4, exit_layers=(1, 2)).eval()
        ch = torch.randint(4, 40, (2, 30))
        pc = torch.full((2, 30), 4)
        with torch.no_grad():
            mem = m.encode_text(ch, pc, None)
            short = torch.zeros(2, 30, dtype=torch.bool)
            short[:, 10:] = True
            long = torch.zeros(2, 30, dtype=torch.bool)
            long[:, 25:] = True
            ds = m.predict_duration(mem, short)
            dl = m.predict_duration(mem, long)
        if not bool(torch.all(dl > ds)):
            raise RuntimeError(
                f"longer text did not predict more frames: {ds.tolist()} vs {dl.tolist()}. "
                "The rate-based duration fix is absent."
            )
        return f"10 tokens -> {float(ds[0]):.1f} frames, 25 tokens -> {float(dl[0]):.1f}"

    @check("discovery rejects duration-confounded splits")
    def _confound():
        from adaptts.data.discovery import discover_word_codes, duration_confound

        rng = np.random.default_rng(0)
        D = 48
        c1 = rng.normal(0, 1, D)
        c2 = c1 + 6 * (rng.normal(0, 1, D) / np.linalg.norm(rng.normal(0, 1, D)))
        feats = np.vstack([c1 + 0.4 * rng.normal(0, 1, (30, D)),
                           c2 + 0.4 * rng.normal(0, 1, (30, D))])
        fast_slow = np.concatenate([np.full(30, 0.25), np.full(30, 1.10)])
        neutral = rng.uniform(0.2, 0.4, 60)
        k_conf = discover_word_codes("w", feats, durations=fast_slow, seed=0).n_codes
        k_ok = discover_word_codes("w", feats, durations=neutral, seed=0).n_codes
        if k_conf != 1:
            raise RuntimeError(
                f"a duration-driven split was accepted (k={k_conf}); the fourth gate is absent"
            )
        if k_ok != 2:
            raise RuntimeError(f"a genuine split was rejected (k={k_ok}); the gate is too strict")
        r2 = duration_confound(np.array([0] * 30 + [1] * 30), fast_slow)
        return f"rejects R^2={r2:.2f} split, accepts the duration-neutral one"

    @check("acoustic generation respects its frame ceiling")
    def _frames():
        from adaptts.models.acoustic import AcousticModel

        m = AcousticModel(40, n_quantizers=8, codebook_size=64, d_model=64,
                          n_layers=2, n_heads=4, d_ff=128, text_d_model=32,
                          text_n_layers=1, text_n_heads=4, depth_d_model=32,
                          depth_n_layers=1, depth_n_heads=4, speaker_dim=16,
                          max_codes=4, exit_layers=(1, 2)).eval()
        ch = torch.randint(4, 40, (1, 20))
        pc = torch.full((1, 20), 4)
        sp = torch.zeros(1, 16)
        with torch.no_grad():
            codes, _ = m.generate(ch, pc, sp, None, depth=2, max_frames=7,
                                  min_frames=7, temperature=0.0)
        if codes.shape[1] > 7:
            raise RuntimeError(f"generated {codes.shape[1]} frames past a ceiling of 7")
        return f"ceiling of 7 produced {codes.shape[1]} frames"

    if not args.skip_downloads:
        print()
        print("=" * 70)
        print("PRETRAINED MODELS (each must load without .bin errors)")
        print("=" * 70)

        @check(f"aligner loads: {cfg.align.model_id}")
        def _align():
            from transformers import AutoModelForCTC, AutoProcessor

            proc = AutoProcessor.from_pretrained(cfg.align.model_id)
            AutoModelForCTC.from_pretrained(cfg.align.model_id)
            vocab = proc.tokenizer.get_vocab()
            blank = next((vocab[n] for n in ("<blank>", "<pad>", "<s>") if n in vocab), 0)
            pad = proc.tokenizer.pad_token_id
            note = "" if blank == pad else f" (blank {blank} differs from pad {pad}, handled)"
            return f"vocab {len(vocab)}, blank id {blank}{note}"

        @check(f"span embedder loads: {cfg.spanemb.model_id}")
        def _span():
            from transformers import AutoFeatureExtractor, AutoModel

            AutoFeatureExtractor.from_pretrained(cfg.spanemb.model_id)
            m = AutoModel.from_pretrained(cfg.spanemb.model_id)
            n = m.config.num_hidden_layers
            if max(cfg.spanemb.layers) > n:
                raise RuntimeError(
                    f"config asks for layer {max(cfg.spanemb.layers)} but the model has {n}"
                )
            return f"{n} layers, reading {list(cfg.spanemb.layers)}"

        @check(f"teacher loads with a fast tokenizer: {cfg.teacher.model_id}")
        def _teacher():
            from transformers import AutoModel, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(cfg.teacher.model_id, use_fast=True)
            if not tok.is_fast:
                raise RuntimeError("a slow tokenizer has no word_ids(); the teacher stage needs it")
            m = AutoModel.from_pretrained(cfg.teacher.model_id)
            if m.config.hidden_size != cfg.teacher.hidden_size:
                raise RuntimeError(
                    f"hidden size {m.config.hidden_size} does not match config "
                    f"{cfg.teacher.hidden_size}"
                )
            enc = tok([["انا", "شوفت", "علم"]], is_split_into_words=True,
                      return_tensors="pt")
            wid = enc.word_ids(0)
            if max(w for w in wid if w is not None) != 2:
                raise RuntimeError("word_ids do not line up with the word count")
            return f"fast tokenizer, hidden {m.config.hidden_size}, word alignment correct"

        @check(f"codec loads and round-trips: {cfg.codec_model_id}")
        def _codec():
            from transformers import MimiModel

            m = MimiModel.from_pretrained(cfg.codec_model_id).eval()
            sr = m.config.sampling_rate
            wav = torch.randn(1, 1, sr * 2) * 0.1
            with torch.no_grad():
                enc = m.encode(wav, num_quantizers=cfg.audio.n_quantizers)
                dec = m.decode(enc.audio_codes).audio_values
            c = enc.audio_codes[0].transpose(0, 1)
            if c.shape[1] != cfg.audio.n_quantizers:
                raise RuntimeError(
                    f"codec returned {c.shape[1]} quantizers, config wants "
                    f"{cfg.audio.n_quantizers}"
                )
            return (
                f"{sr} Hz, 2 s -> {c.shape[0]} frames "
                f"({c.shape[0] / 2:.1f} Hz) -> {dec.shape[-1]} samples"
            )

    print()
    print("=" * 70)
    failed = [r for r in RESULTS if not r[0]]
    if failed:
        print(f"{len(failed)} CHECK(S) FAILED - do not start training")
        for _, name, detail in failed:
            print(f"  - {name}: {detail}")
        print("=" * 70)
        sys.exit(1)
    print(f"ALL {len(RESULTS)} CHECKS PASSED - safe to start training")
    print("=" * 70)


if __name__ == "__main__":
    main()
