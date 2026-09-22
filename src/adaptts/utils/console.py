"""Console setup for Arabic output.

Windows consoles default to a legacy code page that cannot encode Arabic, so
printing a word like a homograph raises UnicodeEncodeError. Scripts call
``enable_utf8_console()`` once at startup so reports and plans are printable
everywhere.
"""

from __future__ import annotations

import sys


def enable_utf8_console() -> None:
    """Reconfigure stdout/stderr to UTF-8 where the platform allows it."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # A redirected or closed stream: printing ASCII still works.
            pass
