"""Generate audio from trained checkpoints and report what happened.

    python scripts/smoke_generate.py --config configs/exp0_small.yaml

This is the check that matters after a training run: does the system produce a
waveform, does the context encoder resolve codes, does an override change the
output, and does a shallower exit actually cost less time. It writes wav files
so you can listen, and prints the numbers so you do not have to guess.

Early in training the audio will not be intelligible. That is expected. What
this script verifies is that the machinery is sound: real samples, finite
values, sensible durations, and a measurable difference between depths.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from adaptts.infer.pipeline import AdapTTS, save_wav  # noqa: E402
from adaptts.utils.console import enable_utf8_console  # noqa: E402
from adaptts.utils.logging_utils import log_table, setup_logging  # noqa: E402

enable_utf8_console()
setup_logging()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate audio from checkpoints")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    ap.add_argument("--n", type=int, default=4, help="how many probe sentences")
    args = ap.parse_args()

    tts = AdapTTS.from_checkpoints(args.config, device=args.device)
    cfg = tts.cfg
    out_dir = Path(args.out) if args.out else Path(cfg.paths.sample_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 74)
    print("LOADED")
    print("=" * 74)
    log_table(
        __import__("logging").getLogger("smoke"),
        ["component", "status"],
        [
            ("context encoder", "yes" if tts.context_encoder else "MISSING"),
            ("acoustic model", "yes" if tts.acoustic else "MISSING"),
            ("codec", "yes" if tts.codec else "MISSING"),
            ("discovered words", len(tts.lexicon.entries) if tts.lexicon else 0),
            ("device", args.device),
        ],
    )
    if tts.acoustic is None or tts.codec is None:
        raise SystemExit("no acoustic checkpoint yet; train stage C first")

    # Prefer sentences containing words this corpus actually discovered, so the
    # override demo has something to override.
    amb = tts.lexicon.ambiguous_words if tts.lexicon else []
    probes = json.load(
        open(Path(__file__).resolve().parent.parent / "assets" / "probe_sentences.json",
             encoding="utf-8")
    )["sentences"]
    sentences = [p["text"] for p in probes][: args.n]
    rows = [json.loads(l) for l in open(cfg.paths.manifest_path, encoding="utf-8")]
    for w in amb[:2]:
        hit = next((r["text"] for r in rows if w in r["text"].split()), None)
        if hit:
            sentences.append(hit)

    print()
    print("=" * 74)
    print("GENERATION")
    print("=" * 74)
    results = []
    for i, text in enumerate(sentences):
        try:
            plan = tts.analyze(text)
            t0 = time.perf_counter()
            wav, st = tts.synthesize(plan)
            elapsed = time.perf_counter() - t0
            path = out_dir / f"smoke_{i:02d}.wav"
            save_wav(str(path), wav, st["sample_rate"])
            hard = ", ".join(f"{w.word}={w.code}" for w in plan.hard_words) or "none"
            print()
            print(f"[{i}] {text[:68]}")
            print(f"    ambiguous : {hard}")
            print(f"    difficulty: {plan.sentence_difficulty:.3f} -> depth {st['depth']}")
            print(f"    audio     : {st['audio_seconds']:.2f}s from {st['frames']:.0f} frames")
            print(f"    speed     : {elapsed:.2f}s wall, real-time factor {st['real_time_factor']:.2f}")
            print(f"    samples   : {len(wav)}, peak {np.abs(wav).max():.3f}, "
                  f"finite {bool(np.isfinite(wav).all())}")
            print(f"    wrote     : {path}")
            results.append((text, st, wav))
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}] FAILED on {text[:50]}: {type(exc).__name__}: {exc}")
            raise

    # Override demo: same sentence, each available reading.
    target = next((p for p in results if tts.analyze(p[0]).hard_words), None)
    if target:
        text = target[0]
        plan = tts.analyze(text)
        w = plan.hard_words[0]
        print()
        print("=" * 74)
        print(f"OVERRIDE DEMO on {w.word!r} ({w.n_codes} readings, no diacritics typed)")
        print("=" * 74)
        for code in range(w.n_codes):
            plan.set_code(w.word, code)
            wav, st = tts.synthesize(plan)
            path = out_dir / f"override_{w.word}_{code}.wav"
            save_wav(str(path), wav, st["sample_rate"])
            print(f"  code {code}: {st['audio_seconds']:.2f}s -> {path}")

    # Depth comparison on one sentence.
    if results:
        text = results[-1][0]
        print()
        print("=" * 74)
        print("ADAPTIVE DEPTH: same sentence at each trained exit")
        print("=" * 74)
        print(f"{'depth':>6}{'wall s':>9}{'audio s':>9}{'RTF':>7}")
        print("-" * 31)
        for d in cfg.acoustic.exit_layers:
            plan = tts.analyze(text)
            plan.set_depth(d)
            tts.synthesize(plan)  # warm
            t0 = time.perf_counter()
            wav, st = tts.synthesize(plan)
            el = time.perf_counter() - t0
            save_wav(str(out_dir / f"depth_{d}.wav"), wav, st["sample_rate"])
            print(f"{d:>6}{el:>9.2f}{st['audio_seconds']:>9.2f}{el / max(st['audio_seconds'], 1e-6):>7.2f}")

    print()
    print(f"wrote {len(list(out_dir.glob('*.wav')))} wav files to {out_dir}")
    print()
    print("Audio quality depends entirely on how long the acoustic model trained.")
    print("This script checks the machinery, not the fidelity.")


if __name__ == "__main__":
    main()
