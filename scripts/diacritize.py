"""Stage A0b: diacritize the corpus with CATT-ECA.

This runs the diacritizer once over every transcript and caches the result.
Nothing downstream needs CATT again, and the shipped model never sees a
diacritic: they exist only to label which reading each word occurrence takes,
exactly as the forced aligner exists only to find spans.

CATT lives in its own environment (it has its own torch pin and model code), so
this script is meant to be run with that interpreter and pointed at the CATT
project root:

    python scripts/diacritize.py \
        --config configs/exp1_egyptian.yaml \
        --catt-root /path/to/Marwa-Code-Switching-TTS-V1.2

On Windows with CATT in WSL, run it from WSL against the same cache directory.

Output: ``<cache>/diacritized.jsonl``, one record per utterance with the plain
text and the diacritized text. That file is all the next stage needs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Diacritize the corpus with CATT-ECA")
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--catt-root", required=True,
        help="directory containing the catt_tashkeel package and its checkpoints",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dialect", default="eca", choices=["eca", "msa"])
    ap.add_argument("--limit", type=int, default=0, help="debug: only N utterances")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from adaptts.utils.config import load_config

    cfg = load_config(args.config)
    manifest = Path(cfg.paths.manifest_path)
    if not manifest.exists():
        raise SystemExit(
            f"manifest not found at {manifest}\n"
            "Run scripts/preprocess.py --stage manifest first."
        )
    out_path = Path(cfg.paths.cache_dir) / "diacritized.jsonl"
    if out_path.exists() and not args.force:
        n = sum(1 for _ in open(out_path, encoding="utf-8"))
        print(f"reusing {n} diacritized utterances at {out_path}")
        print("pass --force to redo")
        return

    rows = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} utterances to diacritize")

    # CATT is imported from its own project tree.
    catt_root = os.path.abspath(args.catt_root)
    if not os.path.isdir(os.path.join(catt_root, "catt_tashkeel")):
        raise SystemExit(f"no catt_tashkeel package under {catt_root}")
    sys.path.insert(0, catt_root)
    cwd = os.getcwd()
    # The wrapper resolves checkpoints relative to the working directory.
    os.chdir(catt_root)
    try:
        from catt_tashkeel.catt_tashkeeler import CattTashkeeler

        print("loading CATT ...")
        tashkeeler = CattTashkeeler(
            preprocess=True, postprocess=True, verbose=False, use_onnx=False
        )

        texts = [r["text"] for r in rows]
        done: List[str] = []
        t0 = time.time()
        step = max(args.batch_size * 8, 64)
        for i in range(0, len(texts), step):
            chunk = texts[i : i + step]
            done.extend(
                tashkeeler.shakkel_batch(
                    chunk, batch_size=args.batch_size, tashkeel_type=args.dialect
                )
            )
            elapsed = time.time() - t0
            rate = len(done) / max(elapsed, 1e-6)
            eta = (len(texts) - len(done)) / max(rate, 1e-6)
            print(
                f"  {len(done)}/{len(texts)}  {rate:.0f}/s  eta {eta/60:.1f} min",
                flush=True,
            )
    finally:
        os.chdir(cwd)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r, d in zip(rows, done):
            f.write(
                json.dumps(
                    {"uid": r["uid"], "text": r["text"], "diacritized": d},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(done)} records to {out_path} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
