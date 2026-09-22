"""Organized logging and progress reporting.

Two goals:

* **Readable logs.** Coloured, aligned, with the stage name in front so a long
  notebook scrollback stays legible.
* **Honest progress.** Every loop over the corpus reports through ``tqdm`` with
  a rate and an ETA, so you can tell at minute two whether a stage will take
  ten minutes or four hours.

``tqdm`` writes to stderr and logging writes to stdout by default, which makes
the two fight over the same lines in a notebook. :func:`setup_logging` routes
logging through ``tqdm.write`` so bars and log lines interleave cleanly.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Optional, Sequence, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")

_BAR = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"


class TqdmLoggingHandler(logging.Handler):
    """Emit log records through ``tqdm.write`` so progress bars stay intact."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stdout)
        except Exception:  # noqa: BLE001
            self.handleError(record)


class StageFormatter(logging.Formatter):
    """Compact, aligned formatter: time, level, logger, message."""

    LEVEL_TAGS = {
        logging.DEBUG: "debug",
        logging.INFO: "info ",
        logging.WARNING: "WARN ",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "FATAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        tag = self.LEVEL_TAGS.get(record.levelno, "info ")
        stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        name = record.name.split(".")[-1][:14]
        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return f"{stamp} {tag} {name:<14} {msg}"


def setup_logging(level: int = logging.INFO, quiet_libraries: bool = True) -> None:
    """Install the tqdm-friendly handler. Safe to call more than once."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = TqdmLoggingHandler()
    handler.setFormatter(StageFormatter())
    root.addHandler(handler)
    root.setLevel(level)

    if quiet_libraries:
        # These are chatty at INFO and drown out our own progress.
        for name in (
            "transformers", "datasets", "huggingface_hub", "filelock",
            "urllib3", "numba", "matplotlib", "fsspec", "torch",
        ):
            logging.getLogger(name).setLevel(logging.WARNING)


def progress(
    iterable: Optional[Iterable[T]] = None,
    *,
    desc: str = "",
    total: Optional[int] = None,
    unit: str = "it",
    leave: bool = True,
    disable: bool = False,
) -> Any:
    """A consistently styled ``tqdm`` bar.

    Used everywhere instead of bare ``tqdm`` so every stage looks the same and
    the description column lines up down the notebook.
    """
    return tqdm(
        iterable,
        desc=f"{desc:<26}" if desc else None,
        total=total,
        unit=unit,
        leave=leave,
        disable=disable,
        dynamic_ncols=True,
        bar_format=_BAR,
        smoothing=0.05,
    )


@contextmanager
def stage(name: str, logger: Optional[logging.Logger] = None) -> Iterator[None]:
    """Log the start and end of a pipeline stage, with a wall-clock duration."""
    log = logger or logging.getLogger("adaptts")
    log.info("")
    log.info("=" * 68)
    log.info(name)
    log.info("=" * 68)
    t0 = time.time()
    try:
        yield
    except Exception:
        log.error("%s FAILED after %s", name, format_duration(time.time() - t0))
        raise
    log.info("%s done in %s", name, format_duration(time.time() - t0))


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def log_table(
    logger: logging.Logger,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    widths: Optional[Sequence[int]] = None,
) -> None:
    """Log an aligned table. Much easier to scan than a dict repr."""
    if widths is None:
        widths = [
            max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
            for i, h in enumerate(headers)
        ]
    head = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    logger.info(head)
    logger.info("-" * len(head))
    for row in rows:
        logger.info("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))


def log_config_summary(logger: logging.Logger, cfg: Any) -> None:
    """Print the handful of settings that actually change a run's behaviour."""
    logger.info("configuration")
    log_table(
        logger,
        ["setting", "value"],
        [
            ("experiment", cfg.name),
            ("dataset", cfg.paths.dataset_dir),
            ("cache", cfg.paths.cache_dir),
            ("run dir", cfg.paths.root),
            ("codec", f"{cfg.codec_model_id} ({cfg.audio.n_quantizers} quantizers)"),
            ("teacher", cfg.teacher.model_id),
            ("aligner", cfg.align.model_id),
            ("precision", cfg.train.precision),
            ("batch size", cfg.train.batch_size),
            ("max steps", cfg.optim.max_steps),
        ],
    )
