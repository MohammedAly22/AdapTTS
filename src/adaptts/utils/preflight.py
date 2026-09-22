"""Preflight checks with actionable error messages.

Training runs are long and often unattended. A missing cache file should say
which stage to run, not raise a bare FileNotFoundError from three frames deep.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .config import Config


def check_preprocessing(cfg: Config, *, need_codes: bool, need_teacher: bool) -> None:
    """Verify the caches a training stage depends on, or explain what to run."""
    required: List[Tuple[Path, str, str]] = [
        (Path(cfg.paths.manifest_path), "manifest", "--stage manifest"),
        (Path(cfg.paths.charvocab_path), "character vocabulary", "--stage manifest"),
        (Path(cfg.paths.lexicon_path), "discovered pronunciation codes", "--stage discover"),
    ]
    if need_codes:
        required.append(
            (Path(cfg.paths.codes_dir) / "codes.npy", "Mimi audio codes", "--stage codec")
        )
    if need_teacher:
        required.append(
            (
                Path(cfg.paths.teacher_dir) / "word_hidden.npy",
                "MARBERTv2 teacher cache",
                "--stage teacher",
            )
        )

    missing = [(p, what, stage) for p, what, stage in required if not p.exists()]
    if not missing:
        return

    lines = ["", "Preprocessing is incomplete. Missing:", ""]
    for p, what, stage in missing:
        lines.append(f"  - {what}")
        lines.append(f"      expected at: {p}")
        lines.append(f"      produced by: python scripts/preprocess.py "
                     f"--config <your-config> {stage}")
        lines.append("")
    lines.append("To run every stage in order:")
    lines.append("  python scripts/preprocess.py --config <your-config> --stage all")
    lines.append("")
    raise SystemExit("\n".join(lines))
