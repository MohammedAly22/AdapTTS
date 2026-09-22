"""Arabic text normalization and word tokenization.

This module contains *no pronunciation knowledge*. It performs Unicode-level
character-class normalization only: NFC composition, diacritic stripping,
tatweel removal, digit folding and whitespace collapsing. Nothing here decides
how a word sounds; that is the job of the discovered pronunciation codes.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

# Unicode ranges, not curated word lists.
ARABIC_DIACRITICS = frozenset(
    chr(c)
    for c in (
        list(range(0x064B, 0x0660))  # fathatan .. sukun, superscript alef region
        + list(range(0x0670, 0x0671))  # superscript alef
        + list(range(0x06D6, 0x06ED + 1))  # quranic annotation marks
        + [0x0640]  # tatweel (kashida)
    )
)

# Zero-width and bidi control characters that carry no phonetic content.
INVISIBLE = frozenset("​‌‍‎‏‪‫‬‭‮﻿")

_ARABIC_INDIC = {chr(0x0660 + i): str(i) for i in range(10)}
_EXT_ARABIC_INDIC = {chr(0x06F0 + i): str(i) for i in range(10)}

_WS_RE = re.compile(r"\s+")
# A "word" is a maximal run of letters/digits/marks. Punctuation is separated
# so it becomes its own token and never fuses into a word type.
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def strip_diacritics(text: str) -> str:
    return "".join(ch for ch in text if ch not in ARABIC_DIACRITICS)


def normalize_text(
    text: str,
    *,
    strip_diacritics_flag: bool = True,
    lowercase_latin: bool = True,
    normalize_alef: bool = False,
    normalize_digits: bool = True,
) -> str:
    """Normalize a transcript to the canonical form used everywhere downstream."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")

    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if ch not in INVISIBLE)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc" or ch in "\t\n")

    if strip_diacritics_flag:
        text = strip_diacritics(text)

    if normalize_digits:
        text = "".join(_ARABIC_INDIC.get(ch, _EXT_ARABIC_INDIC.get(ch, ch)) for ch in text)

    if normalize_alef:
        # Folding hamza forms loses orthographic information that can be the
        # only cue distinguishing two readings, so this is off by default.
        for src, dst in (("أ", "ا"), ("إ", "ا"), ("آ", "ا")):
            text = text.replace(src, dst)

    if lowercase_latin:
        text = "".join(ch.lower() if ch.isascii() and ch.isalpha() else ch for ch in text)

    text = _WS_RE.sub(" ", text).strip()
    return text


def word_spans(text: str) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` character offsets of each word in ``text``."""
    return [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def tokenize_words(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Split normalized text into words plus their character offsets."""
    spans = word_spans(text)
    return [text[s:e] for s, e in spans], spans


def is_arabic_word(word: str) -> bool:
    return any("؀" <= ch <= "ۿ" for ch in word)
