"""Stage A0b: diacritize the corpus with CATT-ECA.

This runs the diacritizer once over every transcript and caches the result.
Nothing downstream needs CATT again, and the shipped model never sees a
diacritic: they exist only to label which reading each word occurrence takes,
exactly as the forced aligner exists only to find spans.

CATT needs its own environment (it pins pytorch-lightning and an older torch),
so this script runs under that interpreter while everything else runs under the
adaptts environment. On RunPod:

    conda activate CATT
    python scripts/diacritize.py \
        --config configs/exp1_egyptian.yaml \
        --catt-root /workspace/catt_parent

``--catt-root`` is the directory *containing* the ``catt_tashkeel`` package, not
the package itself.

Only the ECA checkpoint is required. ``CattTashkeeler`` cannot be used because
its constructor loads the MSA checkpoint unconditionally, so this script builds
the ECA model directly from the same classes. Egyptian is the only dialect this
project needs, and the MSA weights are a 74 MB download that would never be
called.

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


def load_eca_model(catt_root: str, verbose: bool = False):
    """Build the ECA diacritizer without touching the MSA checkpoint.

    Returns ``(model, preprocessor, postprocessor)``. The wrapper class in the
    CATT tree instantiates both dialects in ``__init__`` and calls
    ``load_state_dict`` on each, so importing it raises ``FileNotFoundError``
    when only ECA weights are present. The pieces below are the same classes
    the wrapper would have used, assembled for one dialect.
    """
    pkg = os.path.join(catt_root, "catt_tashkeel")
    if not os.path.isdir(pkg):
        raise SystemExit(
            f"no catt_tashkeel package under {catt_root}\n"
            f"--catt-root must be the directory containing catt_tashkeel/, "
            f"not catt_tashkeel/ itself."
        )

    ckpt = os.path.join(pkg, "checkpoints", "eca_model_weights.pt")
    if not os.path.isfile(ckpt):
        raise SystemExit(
            f"ECA checkpoint missing at {ckpt}\n"
            "This is the one weights file the pipeline needs."
        )

    if catt_root not in sys.path:
        sys.path.insert(0, catt_root)

    import torch
    from catt_tashkeel.model import TashkeelModel
    from catt_tashkeel.postprocessor import PostProcessor
    from catt_tashkeel.preprocessing.eca_preprocessor import ECAPreProcessor
    from catt_tashkeel.tokenizer import TashkeelTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tokenizer = TashkeelTokenizer()
    # These hyperparameters come from the wrapper; they must match the
    # checkpoint or load_state_dict fails loudly, which is the desired
    # behaviour rather than silently loading a mismatched model.
    model = TashkeelModel(
        tokenizer=tokenizer,
        max_seq_len=1024,
        d_model=512,
        n_layers=6,
        n_heads=16,
        learnable_pos_emb=True,
    )
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model = model.eval().to(device)

    return model, ECAPreProcessor(verbose=verbose), PostProcessor(verbose=verbose), tokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description="Diacritize the corpus with CATT-ECA")
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--catt-root", default="",
        help="directory containing the catt_tashkeel package. Falls back to "
             "$CATT_ROOT, then to paths.catt_root in the config.",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N utterances")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from adaptts.utils.config import load_config

    cfg = load_config(args.config)

    # Flag, then environment, then config. The notebook sets the config value,
    # so the common case needs no argument at all.
    catt_root = args.catt_root or os.environ.get("CATT_ROOT", "") or cfg.paths.catt_root
    if not catt_root:
        raise SystemExit(
            "No CATT location given. Set paths.catt_root in the config, or "
            "export CATT_ROOT, or pass --catt-root.\n"
            "It is the directory *containing* catt_tashkeel/."
        )
    manifest = Path(cfg.paths.manifest_path)
    if not manifest.exists():
        raise SystemExit(
            f"manifest not found at {manifest}\n"
            "Run scripts/preprocess.py --stage manifest first."
        )

    out_path = Path(cfg.paths.diacritized_path)
    if out_path.exists() and not args.force:
        n = sum(1 for _ in open(out_path, encoding="utf-8"))
        print(f"reusing {n} diacritized utterances at {out_path}")
        print("pass --force to redo")
        return

    rows = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} utterances to diacritize")

    model, pre, post, tokenizer = load_eca_model(os.path.abspath(catt_root))
    print("CATT-ECA loaded")

    texts = [r["text"] for r in rows]
    done: List[str] = []
    t0 = time.time()
    step = max(args.batch_size * 8, 64)

    for i in range(0, len(texts), step):
        chunk = texts[i : i + step]
        # Mirror the wrapper's ECA path: strip any existing marks, preprocess,
        # diacritize, postprocess.
        cleaned = [tokenizer.remove_tashkeel(t) for t in chunk]
        processed = [pre.process_text(t, verbose=False) for t in cleaned]
        out = model.do_tashkeel_batch(
            processed, batch_size=args.batch_size, verbose=False
        )
        done.extend(post.process(t) for t in out)

        elapsed = time.time() - t0
        rate = len(done) / max(elapsed, 1e-6)
        eta = (len(texts) - len(done)) / max(rate, 1e-6)
        print(
            f"  {len(done)}/{len(texts)}  {rate:.0f}/s  eta {eta/60:.1f} min",
            flush=True,
        )

    if len(done) != len(rows):
        raise SystemExit(
            f"diacritizer returned {len(done)} texts for {len(rows)} inputs; "
            "refusing to write a misaligned cache"
        )

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
