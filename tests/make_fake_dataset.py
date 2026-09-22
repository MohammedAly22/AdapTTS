"""Build a small synthetic dataset for integration testing.

Generates real WAV files and a manifest in the Masri-100h layout, so the actual
preprocessing and training scripts can be exercised end to end without
downloading 12 GB. The audio is synthetic, so the resulting model is not
useful, but every code path is.

    python tests/make_fake_dataset.py --out /tmp/fake --n 48
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from adaptts.utils.console import enable_utf8_console  # noqa: E402

FLAG = [
    "انا شوفت علم مصر بيرفرف",
    "علم مصر لونه احمر وابيض واسود",
    "رفعوا علم النادي فوق المدرج",
    "الولد شال علم كبير في ايده",
]
SCIENCE = [
    "علم الفيزيا من اهم العلوم البشرية",
    "علم الرياضيات صعب بس مفيد جدا",
    "هو بيدرس علم النفس في الجامعة",
    "علم الاحياء بيدرس الكائنات الحية",
]
PLAIN = [
    "الجو النهارده حلو جدا و الشمس طالعة",
    "احنا رايحين السوق بكرة الصبح",
    "الاكل كان لذيذ والناس كانت مبسوطة",
    "القطر اتاخر ساعة بسبب الزحمة",
]


def synth_wav(seconds: float, sr: int, f0: float, seed: int) -> np.ndarray:
    """A crude voiced-speech-like signal: harmonic stack plus shaped noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    sig = np.zeros_like(t)
    for h in range(1, 12):
        sig += (1.0 / h) * np.sin(2 * math.pi * f0 * h * t + rng.uniform(0, 2 * math.pi))
    env = 0.5 * (1 - np.cos(2 * math.pi * np.clip(t / seconds, 0, 1)))
    sig = sig * env + 0.02 * rng.standard_normal(t.shape)
    peak = float(np.abs(sig).max()) or 1.0
    return (0.6 * sig / peak).astype(np.float32)


def main() -> None:
    enable_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--sr", type=int, default=24000)
    args = ap.parse_args()

    import soundfile as sf

    root = Path(args.out)
    (root / "clips" / "vid0").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)

    rows = []
    pool = [(t, "flag") for t in FLAG] + [(t, "sci") for t in SCIENCE] + [(t, "plain") for t in PLAIN]
    rng = random.Random(0)

    for i in range(args.n):
        text, kind = pool[i % len(pool)]
        # The two readings get distinct pitch, so discovery has a real acoustic
        # difference to find, mirroring what it would find in real speech.
        f0 = {"flag": 130.0, "sci": 190.0, "plain": 160.0}[kind]
        dur = 2.0 + 0.05 * len(text) + rng.uniform(-0.2, 0.2)
        dur = max(1.5, min(dur, 8.0))
        wav = synth_wav(dur, args.sr, f0 + rng.uniform(-4, 4), seed=i)

        name = f"vid0_{i:04d}"
        rel = f"clips/vid0/{name}.wav"
        sf.write(root / rel, wav, args.sr)
        (root / "clips" / "vid0" / f"{name}.txt").write_text(text, encoding="utf-8")

        split = "train" if i < int(args.n * 0.8) else ("dev" if i < int(args.n * 0.9) else "test")
        rows.append({
            "file_name": rel, "id": name, "text": text,
            "duration": round(len(wav) / args.sr, 3), "confidence": 0.99,
            "channel": "fake", "video_id": "vid0", "promo": False, "split": split,
        })

    with open(root / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for split in ("train", "dev", "test"):
        sel = [r for r in rows if r["split"] == split]
        with open(root / "metadata" / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in sel:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(root / "metadata" / "all.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "dev", "test")}
    print(f"wrote {len(rows)} clips to {root}  splits={counts}")


if __name__ == "__main__":
    main()
