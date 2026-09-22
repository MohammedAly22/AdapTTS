"""Egyptian Arabic text normalization for TTS.

Converts anything a real transcript can contain into speakable Egyptian Arabic
words: numbers, dates, times, currency, percentages, phone numbers, URLs,
emails, addresses, abbreviations and Latin punctuation.

## The number rule

Egyptian Arabic does **not** insert a linking waw between magnitude groups the
way Modern Standard Arabic does. The waw appears in exactly one place: between
the unit and the ten inside a two-digit group.

    2024 -> الفين اربعة و عشرين          (not: الفين و اربعة و عشرين)
     876 -> تمنمية ستة و سبعين            (not: تمنمية و ستة و سبعين)
    1986 -> الف تسعمية ستة و تمانين       (not: الف و تسعمية و ستة و تمانين)

So the waw is a property of the tens group, not a separator between groups.
That is why :func:`_two_digits` is the only place a waw is emitted.

## Scope and philosophy

This module is a *text* normalizer. It decides how to say "2024" or "د." in
Egyptian Arabic. It makes no pronunciation choices that context could change:
resolving a homograph such as علم is the job of the discovered pronunciation
codes, not of this file. The two concerns stay separate on purpose.

Run it directly to see the test suite::

    python -m adaptts.text.egyptian
    python src/adaptts/text/egyptian.py
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Character-level tables
# ---------------------------------------------------------------------------

ARABIC_DIACRITICS = "".join(
    chr(c)
    for c in list(range(0x064B, 0x0653))
    + [0x0654, 0x0655, 0x0656, 0x0657, 0x0658, 0x0670]
    + list(range(0x06D6, 0x06ED + 1))
)
TATWEEL = "ـ"

_ARABIC_INDIC = {chr(0x0660 + i): str(i) for i in range(10)}
_EXT_ARABIC_INDIC = {chr(0x06F0 + i): str(i) for i in range(10)}
_DIGIT_MAP = {**_ARABIC_INDIC, **_EXT_ARABIC_INDIC}

INVISIBLE = "​‌‍‎‏‪‫‬‭‮﻿­"

# Punctuation that should be read as a pause, not spoken.
PUNCT_TO_SPACE = "«»\"'`´“”‘’()[]{}<>|\\/_*#^~+=@&"
SENTENCE_PUNCT = {"،": "،", ",": "،", ";": "،", "؛": "،", ":": "،",
                  ".": ".", "؟": "؟", "?": "؟", "!": "!", "…": "."}


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

# Egyptian colloquial forms, masculine/counting shape used when reading digits.
ONES = [
    "", "واحد", "اتنين", "تلاتة", "اربعة", "خمسة", "ستة", "سبعة", "تمانية", "تسعة",
]
ONES_FEM = [
    "", "واحدة", "اتنين", "تلاتة", "اربعة", "خمسة", "ستة", "سبعة", "تمانية", "تسعة",
]
TEENS = [
    "عشرة", "حداشر", "اتناشر", "تلاتاشر", "اربعتاشر", "خمستاشر",
    "ستاشر", "سبعتاشر", "تمنتاشر", "تسعتاشر",
]
TENS = [
    "", "", "عشرين", "تلاتين", "اربعين", "خمسين",
    "ستين", "سبعين", "تمانين", "تسعين",
]
# Hundreds are single fused words in Egyptian Arabic.
HUNDREDS = [
    "", "مية", "متين", "تلتمية", "اربعمية", "خمسمية",
    "ستمية", "سبعمية", "تمنمية", "تسعمية",
]

# Scale words: (singular, dual, plural-with-small-count, bare-plural)
SCALES: List[Tuple[int, str, str, str]] = [
    (10**12, "تريليون", "تريليونين", "تريليون"),
    (10**9, "مليار", "مليارين", "مليار"),
    (10**6, "مليون", "مليونين", "ملايين"),
    (10**3, "الف", "الفين", "الاف"),
]


def _two_digits(n: int, feminine: bool = False) -> str:
    """Read 0..99.

    This is the **only** function that emits a linking waw, and it emits it only
    between a unit and a ten. Everything above 99 concatenates without a waw,
    which is what distinguishes Egyptian Arabic from MSA here.
    """
    if not 0 <= n <= 99:
        raise ValueError(f"_two_digits expects 0..99, got {n}")
    ones = ONES_FEM if feminine else ONES
    if n == 0:
        return ""
    if n < 10:
        return ones[n]
    if n < 20:
        return TEENS[n - 10]
    unit, ten = n % 10, n // 10
    if unit == 0:
        return TENS[ten]
    # The one and only waw: "ستة و سبعين", "اربعة و عشرين".
    return f"{ones[unit]} و {TENS[ten]}"


def _three_digits(n: int, feminine: bool = False) -> str:
    """Read 0..999. Hundreds and the remainder are juxtaposed, with no waw."""
    if not 0 <= n <= 999:
        raise ValueError(f"_three_digits expects 0..999, got {n}")
    if n == 0:
        return ""
    hundreds, rest = n // 100, n % 100
    parts: List[str] = []
    if hundreds:
        parts.append(HUNDREDS[hundreds])
    if rest:
        parts.append(_two_digits(rest, feminine))
    # No waw here: 876 -> "تمنمية ستة و سبعين".
    return " ".join(parts)


def _scale_word(count: int, singular: str, dual: str, plural: str) -> str:
    """Pick the right form of a scale word for the group value.

    Arabic number agreement: 1 takes the singular, 2 the dual, 3..10 the plural,
    and 11+ returns to the singular.
    """
    if count == 1:
        return singular
    if count == 2:
        return dual
    if 3 <= count <= 10:
        return f"{_three_digits(count)} {plural}"
    return f"{_three_digits(count)} {singular}"


def number_to_words(n: int, feminine: bool = False) -> str:
    """Convert a non-negative integer to Egyptian Arabic words.

    >>> number_to_words(2024)
    'الفين اربعة و عشرين'
    >>> number_to_words(876)
    'تمنمية ستة و سبعين'
    >>> number_to_words(1986)
    'الف تسعمية ستة و تمانين'
    """
    if n < 0:
        return f"سالب {number_to_words(-n, feminine)}"
    if n == 0:
        return "صفر"

    parts: List[str] = []
    remaining = n
    for value, singular, dual, plural in SCALES:
        if remaining >= value:
            count = remaining // value
            remaining %= value
            parts.append(_scale_word(count, singular, dual, plural))

    if remaining:
        parts.append(_three_digits(remaining, feminine))

    # Groups are juxtaposed with no linking waw. This is the Egyptian pattern.
    return " ".join(p for p in parts if p)


def decimal_to_words(text: str) -> str:
    """Read a decimal number: ``3.5`` -> ``تلاتة فاصلة خمسة``."""
    if "." in text:
        whole, frac = text.split(".", 1)
    elif "٫" in text:
        whole, frac = text.split("٫", 1)
    else:
        return number_to_words(int(text))
    whole_words = number_to_words(int(whole)) if whole else "صفر"
    # Fractional digits are read one by one, as in speech.
    frac_words = " ".join(
        ONES[int(d)] if d != "0" else "صفر" for d in frac if d.isdigit()
    )
    return f"{whole_words} فاصلة {frac_words}"


def ordinal_to_words(n: int, feminine: bool = False) -> str:
    """Read an ordinal: 1st..10th have dedicated words, above that use cardinals."""
    masc = ["", "الاول", "التاني", "التالت", "الرابع", "الخامس",
            "السادس", "السابع", "التامن", "التاسع", "العاشر"]
    fem = ["", "الاولى", "التانية", "التالتة", "الرابعة", "الخامسة",
           "السادسة", "السابعة", "التامنة", "التاسعة", "العاشرة"]
    table = fem if feminine else masc
    if 1 <= n <= 10:
        return table[n]
    return number_to_words(n, feminine)


def digits_one_by_one(s: str) -> str:
    """Read a digit string as separate digits, for phone numbers and codes."""
    out = []
    for ch in s:
        if ch.isdigit():
            out.append("صفر" if ch == "0" else ONES[int(ch)])
    return " ".join(out)


# ---------------------------------------------------------------------------
# Dates and times
# ---------------------------------------------------------------------------

MONTHS_NUM = {
    1: "يناير", 2: "فبراير", 3: "مارس", 4: "ابريل", 5: "مايو", 6: "يونيو",
    7: "يوليو", 8: "اغسطس", 9: "سبتمبر", 10: "اكتوبر", 11: "نوفمبر", 12: "ديسمبر",
}


def date_to_words(day: int, month: int, year: Optional[int] = None) -> str:
    """Read a calendar date in the Egyptian spoken order: day, month, year."""
    if not 1 <= month <= 12:
        raise ValueError(f"month out of range: {month}")
    parts = [number_to_words(day), MONTHS_NUM[month]]
    if year is not None:
        parts.append(number_to_words(year))
    return " ".join(parts)


def time_to_words(hour: int, minute: int = 0) -> str:
    """Read a clock time using the Egyptian fractional expressions."""
    hour_12 = hour % 12 or 12
    hour_names = {
        1: "الواحدة", 2: "التانية", 3: "التالتة", 4: "الرابعة", 5: "الخامسة",
        6: "السادسة", 7: "السابعة", 8: "التامنة", 9: "التاسعة", 10: "العاشرة",
        11: "الحداشر", 12: "التناشر",
    }
    base = f"الساعة {hour_names[hour_12]}"
    if minute == 0:
        return base
    if minute == 15:
        return f"{base} و ربع"
    if minute == 30:
        return f"{base} و نص"
    if minute == 45:
        nxt = hour_names[(hour_12 % 12) + 1]
        return f"الساعة {nxt} الا ربع"
    if minute == 20:
        return f"{base} و تلت"
    if minute == 40:
        nxt = hour_names[(hour_12 % 12) + 1]
        return f"الساعة {nxt} الا تلت"
    return f"{base} و {number_to_words(minute)} دقيقة"


# ---------------------------------------------------------------------------
# Lookup tables: abbreviations, currency, units, symbols
# ---------------------------------------------------------------------------

ABBREVIATIONS: Dict[str, str] = {
    "د.": "دكتور",
    "د/": "دكتور",
    "أ.د": "استاذ دكتور",
    "ا.د": "استاذ دكتور",
    "م.": "مهندس",
    "م/": "مهندس",
    "أ.": "استاذ",
    "ا.": "استاذ",
    "ص.ب": "صندوق بريد",
    "ش.": "شارع",
    "ق.م": "قبل الميلاد",
    "ب.م": "بعد الميلاد",
    "الخ": "الى اخره",
    "إلخ": "الى اخره",
    "ج.م": "جنيه مصري",
    "كج": "كيلو جرام",
    "كم": "كيلو متر",
    "سم": "سنتي متر",
    "مم": "ملي متر",
    "هـ": "هجري",
}

LATIN_ABBREVIATIONS: Dict[str, str] = {
    "dr": "دكتور",
    "prof": "بروفيسور",
    "mr": "مستر",
    "mrs": "مدام",
    "eng": "مهندس",
    "st": "شارع",
    "am": "صباحا",
    "pm": "مساء",
    "km": "كيلو متر",
    "kg": "كيلو جرام",
    "cm": "سنتي متر",
    "mm": "ملي متر",
    "gb": "جيجا بايت",
    "mb": "ميجا بايت",
    "kb": "كيلو بايت",
    "tv": "تي في",
    "pc": "بي سي",
    "ok": "او كيه",
    "ceo": "سي اي او",
    "usa": "امريكا",
    "uk": "بريطانيا",
    "uae": "الامارات",
}

CURRENCY: Dict[str, Tuple[str, str]] = {
    # symbol -> (singular/plural unit, subunit)
    "ج.م": ("جنيه", "قرش"),
    "جنيه": ("جنيه", "قرش"),
    "$": ("دولار", "سنت"),
    "USD": ("دولار", "سنت"),
    "€": ("يورو", "سنت"),
    "EUR": ("يورو", "سنت"),
    "£": ("جنيه استرليني", "بنس"),
    "GBP": ("جنيه استرليني", "بنس"),
    "ر.س": ("ريال", "هللة"),
    "SAR": ("ريال", "هللة"),
    "د.إ": ("درهم", "فلس"),
    "AED": ("درهم", "فلس"),
}

SYMBOLS: Dict[str, str] = {
    "%": "في المية",
    "٪": "في المية",
    "&": "و",
    "@": "ات",
    "+": "زائد",
    "=": "يساوي",
    "°": "درجة",
    "×": "في",
    "÷": "على",
    "©": "حقوق النشر",
    "™": "علامة تجارية",
}

# Latin letters spelled out, for acronyms and URLs.
LATIN_LETTERS: Dict[str, str] = {
    "a": "ايه", "b": "بي", "c": "سي", "d": "دي", "e": "اي", "f": "اف",
    "g": "جي", "h": "اتش", "i": "اي", "j": "جي", "k": "كيه", "l": "ال",
    "m": "ام", "n": "ان", "o": "او", "p": "بي", "q": "كيو", "r": "ار",
    "s": "اس", "t": "تي", "u": "يو", "v": "في", "w": "دبليو",
    "x": "اكس", "y": "واي", "z": "زد",
}

DOMAIN_WORDS: Dict[str, str] = {
    "com": "كوم", "net": "نت", "org": "اورج", "edu": "ادو", "gov": "جوف",
    "eg": "اي جي", "io": "اي او", "ai": "ايه اي", "co": "كو",
}

COMMON_SITES: Dict[str, str] = {
    "google": "جوجل", "youtube": "يوتيوب", "facebook": "فيسبوك",
    "twitter": "تويتر", "instagram": "انستجرام", "whatsapp": "واتساب",
    "gmail": "جيميل", "yahoo": "ياهو", "hotmail": "هوتميل",
    "outlook": "اوتلوك", "linkedin": "لينكد ان", "github": "جيت هاب",
    "tiktok": "تيك توك", "telegram": "تليجرام", "netflix": "نتفليكس",
    "amazon": "امازون", "wikipedia": "ويكيبيديا", "huggingface": "هجينج فيس",
}


# ---------------------------------------------------------------------------
# Regex patterns, ordered by priority
# ---------------------------------------------------------------------------

RE_URL = re.compile(
    r"\b(?:https?://|www\.)[^\s،,؛;]+|\b[\w-]+\.(?:com|net|org|edu|gov|eg|io|ai|co)\b(?:/[^\s،,؛;]*)?",
    re.IGNORECASE,
)
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
RE_TIME = re.compile(r"\b([01]?\d|2[0-3])[:٫:]([0-5]\d)(?:\s*(am|pm|ص|م))?\b", re.IGNORECASE)
RE_DATE_DMY = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b")
RE_DATE_MY = re.compile(r"\b(\d{1,2})[/\-](\d{4})\b")
RE_PHONE = re.compile(r"(?<![\d])(?:\+?\d[\d\s-]{7,16}\d)(?![\d])")
RE_PERCENT = re.compile(r"(\d+(?:[.,٫]\d+)?)\s*[%٪]")
RE_CURRENCY_PRE = re.compile(r"([$€£])\s*(\d+(?:[.,٫]\d+)?)")
RE_CURRENCY_POST = re.compile(
    r"(\d+(?:[.,٫]\d+)?)\s*(ج\.م|جنيه|ر\.س|د\.إ|USD|EUR|GBP|SAR|AED|\$|€|£)",
    re.IGNORECASE,
)
RE_RANGE = re.compile(r"\b(\d+)\s*[-–]\s*(\d+)\b")
RE_FRACTION = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")
RE_ORDINAL_AR = re.compile(r"\bال(\d+)\b")
RE_THOUSANDS_SEP = re.compile(r"\b\d{1,3}(?:[,٬]\d{3})+\b")
RE_DECIMAL = re.compile(r"\b\d+[.,٫]\d+\b")
RE_INTEGER = re.compile(r"\d+")
RE_LATIN_WORD = re.compile(r"\b[A-Za-z][A-Za-z.]*\b")
RE_REPEAT = re.compile(r"(.)\1{3,}")
RE_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Individual replacement handlers
# ---------------------------------------------------------------------------


def _say_latin_token(token: str) -> str:
    """Speak a Latin token: known word, known abbreviation, or spelled out."""
    low = token.lower().strip(".")
    if not low:
        return ""
    if low in COMMON_SITES:
        return COMMON_SITES[low]
    if low in LATIN_ABBREVIATIONS:
        return LATIN_ABBREVIATIONS[low]
    if low in DOMAIN_WORDS:
        return DOMAIN_WORDS[low]
    # An all-caps short token is an acronym: spell it.
    if token.isupper() and len(token) <= 5:
        return " ".join(LATIN_LETTERS.get(c, c) for c in low if c.isalpha())
    if len(low) <= 3:
        return " ".join(LATIN_LETTERS.get(c, c) for c in low if c.isalpha())
    # Unknown longer word: spell it rather than guess a transliteration, since a
    # wrong guess is worse than letters for a TTS front end.
    return " ".join(LATIN_LETTERS.get(c, c) for c in low if c.isalpha())


def _expand_url(m: re.Match) -> str:
    url = m.group(0)
    body = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
    body = re.sub(r"^www\.", "دبليو دبليو دبليو نقطة ", body, flags=re.IGNORECASE)
    path = ""
    if "/" in body:
        body, _, path = body.partition("/")
    parts = [p for p in body.split(".") if p]
    said = []
    for i, p in enumerate(parts):
        if i:
            said.append("نقطة")
        said.append(_say_latin_token(p) if re.match(r"^[A-Za-z]", p) else p)
    out = " ".join(said)
    if path.strip("/"):
        out += " سلاش " + " ".join(
            _say_latin_token(seg) for seg in re.split(r"[/\-_]", path) if seg
        )
    return f" {out} "


def _expand_email(m: re.Match) -> str:
    user, _, domain = m.group(0).partition("@")
    user_said = " ".join(
        _say_latin_token(p) if p else "" for p in re.split(r"[._+-]", user) if p
    )
    dom_parts = [p for p in domain.split(".") if p]
    dom_said = []
    for i, p in enumerate(dom_parts):
        if i:
            dom_said.append("نقطة")
        dom_said.append(_say_latin_token(p))
    return f" {user_said} ات {' '.join(dom_said)} "


def _expand_time(m: re.Match) -> str:
    hour, minute = int(m.group(1)), int(m.group(2))
    suffix = (m.group(3) or "").lower()
    out = time_to_words(hour, minute)
    # The reading already begins with الساعة. If the sentence just said it,
    # drop ours instead of stuttering.
    before = m.string[: m.start()].rstrip()
    if before.endswith("الساعة") and out.startswith("الساعة "):
        out = out[len("الساعة "):]
    if suffix in ("am", "ص"):
        out += " صباحا"
    elif suffix in ("pm", "م"):
        out += " مساء"
    return f" {out} "


def _expand_date_dmy(m: re.Match) -> str:
    a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if c < 100:
        c += 2000 if c < 50 else 1900
    if not 1 <= b <= 12:
        # Not a valid month, so treat it as a plain number sequence.
        return f" {number_to_words(a)} {number_to_words(b)} {number_to_words(c)} "
    return f" {date_to_words(a, b, c)} "


def _expand_date_my(m: re.Match) -> str:
    month, year = int(m.group(1)), int(m.group(2))
    if not 1 <= month <= 12:
        return f" {number_to_words(month)} {number_to_words(year)} "
    return f" {MONTHS_NUM[month]} {number_to_words(year)} "


def _expand_phone(m: re.Match) -> str:
    raw = m.group(0)
    digits = re.sub(r"\D", "", raw)
    # Only treat it as a phone number if it is long enough to be one.
    if not 9 <= len(digits) <= 15:
        return raw
    prefix = "زائد " if raw.strip().startswith("+") else ""
    return f" {prefix}{digits_one_by_one(digits)} "


def _expand_percent(m: re.Match) -> str:
    num = m.group(1).replace(",", ".").replace("٫", ".")
    said = decimal_to_words(num) if "." in num else number_to_words(int(num))
    return f" {said} في المية "


def _expand_currency_pre(m: re.Match) -> str:
    sym, num = m.group(1), m.group(2).replace(",", ".").replace("٫", ".")
    unit = CURRENCY.get(sym, ("جنيه", "قرش"))[0]
    said = decimal_to_words(num) if "." in num else number_to_words(int(num))
    return f" {said} {unit} "


def _expand_currency_post(m: re.Match) -> str:
    num, sym = m.group(1).replace(",", ".").replace("٫", "."), m.group(2)
    key = sym if sym in CURRENCY else sym.upper()
    unit = CURRENCY.get(key, ("جنيه", "قرش"))[0]
    said = decimal_to_words(num) if "." in num else number_to_words(int(num))
    return f" {said} {unit} "


def _expand_range(m: re.Match) -> str:
    return f" من {number_to_words(int(m.group(1)))} لـ {number_to_words(int(m.group(2)))} "


def _expand_fraction(m: re.Match) -> str:
    num, den = int(m.group(1)), int(m.group(2))
    singular = {2: "نص", 3: "تلت", 4: "ربع", 5: "خمس", 6: "سدس",
                7: "سبع", 8: "تمن", 9: "تسع", 10: "عشر"}
    dual = {2: "نصين", 3: "تلتين", 4: "ربعين", 5: "خمسين", 6: "سدسين",
            7: "سبعين", 8: "تمنين", 9: "تسعين", 10: "عشرين"}
    plural = {2: "انصاف", 3: "تلات", 4: "ارباع", 5: "اخماس", 6: "اسداس",
              7: "اسباع", 8: "اتمان", 9: "اتساع", 10: "اعشار"}
    if den not in singular:
        return f" {number_to_words(num)} على {number_to_words(den)} "
    if num == 1:
        return f" {singular[den]} "
    if num == 2:
        return f" {dual[den]} "
    # 3..10 of a fraction take the broken plural: 3/4 -> "تلاتة ارباع".
    if 3 <= num <= 10:
        return f" {number_to_words(num)} {plural[den]} "
    return f" {number_to_words(num)} على {number_to_words(den)} "


def _expand_ordinal_ar(m: re.Match) -> str:
    return f" {ordinal_to_words(int(m.group(1)))} "


def _expand_decimal(m: re.Match) -> str:
    return f" {decimal_to_words(m.group(0).replace(',', '.').replace('٫', '.'))} "


def _expand_integer(m: re.Match) -> str:
    return f" {number_to_words(int(m.group(0)))} "


def _expand_latin(m: re.Match) -> str:
    return f" {_say_latin_token(m.group(0))} "


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def strip_diacritics(text: str) -> str:
    table = str.maketrans("", "", ARABIC_DIACRITICS + TATWEEL)
    return text.translate(table)


def normalize_characters(text: str, *, strip_marks: bool = True) -> str:
    """Unicode-level cleanup: composition, invisibles, digits, diacritics."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(str.maketrans("", "", INVISIBLE))
    if strip_marks:
        text = strip_diacritics(text)
    text = "".join(_DIGIT_MAP.get(ch, ch) for ch in text)
    return text


# Arabic has no case, so a "word boundary" here means: not flanked by another
# Arabic letter. A plain str.replace would rewrite الخصم into "الى اخره صم" by
# matching الخ inside the word.
_AR_LETTER = "\u0621-\u064a"


def _abbrev_pattern(abbr: str) -> "re.Pattern":
    esc = re.escape(abbr)
    # An abbreviation ending in a dot or slash carries its own right boundary;
    # one ending in a letter needs an explicit guard.
    right = "" if abbr[-1] in "./" else "(?![" + _AR_LETTER + "])"
    return re.compile("(?<![" + _AR_LETTER + "])" + esc + right)


_ABBREV_PATTERNS = [
    (_abbrev_pattern(a), ABBREVIATIONS[a])
    for a in sorted(ABBREVIATIONS, key=len, reverse=True)
]


def expand_abbreviations(text: str) -> str:
    """Expand Arabic abbreviations, longest first, respecting word boundaries."""
    for pattern, replacement in _ABBREV_PATTERNS:
        text = pattern.sub(" " + replacement + " ", text)
    return text


def expand_symbols(text: str) -> str:
    for sym, word in SYMBOLS.items():
        if sym in text:
            text = text.replace(sym, f" {word} ")
    return text


def normalize_egyptian(
    text: str,
    *,
    expand_numbers: bool = True,
    expand_latin: bool = True,
    keep_punctuation: bool = True,
    strip_marks: bool = True,
) -> str:
    """Normalize one line of Egyptian Arabic into speakable words.

    Order matters. Structured patterns (URLs, emails, dates) are consumed before
    the bare-number rule, or a date like 12/5/2024 would be read as three
    unrelated integers.
    """
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    if not text.strip():
        return ""

    text = normalize_characters(text, strip_marks=strip_marks)
    text = RE_REPEAT.sub(r"\1\1", text)  # "جااااامد" -> "جاامد"

    # Strip digit-grouping separators before any rule can read them as a
    # decimal point: 2,500,000 is two and a half million, not "two point five".
    text = RE_THOUSANDS_SEP.sub(
        lambda m: m.group(0).replace(",", "").replace("٬", ""), text
    )

    # 1. Structured spans first: they contain digits and dots that later rules
    #    would otherwise shred.
    text = RE_URL.sub(_expand_url, text)
    text = RE_EMAIL.sub(_expand_email, text)
    text = RE_TIME.sub(_expand_time, text)
    text = RE_DATE_DMY.sub(_expand_date_dmy, text)
    text = RE_DATE_MY.sub(_expand_date_my, text)

    # 2. Abbreviations, before punctuation is stripped (many contain a dot).
    text = expand_abbreviations(text)

    # 3. Quantities.
    if expand_numbers:
        text = RE_PERCENT.sub(_expand_percent, text)
        text = RE_CURRENCY_PRE.sub(_expand_currency_pre, text)
        text = RE_CURRENCY_POST.sub(_expand_currency_post, text)
        text = RE_PHONE.sub(_expand_phone, text)
        text = RE_FRACTION.sub(_expand_fraction, text)
        text = RE_RANGE.sub(_expand_range, text)
        text = RE_ORDINAL_AR.sub(_expand_ordinal_ar, text)
        text = RE_DECIMAL.sub(_expand_decimal, text)

    text = expand_symbols(text)

    if expand_numbers:
        text = RE_INTEGER.sub(_expand_integer, text)

    if expand_latin:
        text = RE_LATIN_WORD.sub(_expand_latin, text)

    # 4. Punctuation.
    text = text.translate(str.maketrans(PUNCT_TO_SPACE, " " * len(PUNCT_TO_SPACE)))
    if keep_punctuation:
        for src, dst in SENTENCE_PUNCT.items():
            text = text.replace(src, f" {dst} ")
    else:
        text = text.translate(str.maketrans("", "", "".join(SENTENCE_PUNCT)))

    # 5. Drop anything still unspeakable, then tidy whitespace.
    text = "".join(
        ch for ch in text
        if ch.isspace() or "؀" <= ch <= "ۿ" or ch in "،.؟!"
    )
    text = RE_WS.sub(" ", text).strip()
    # Collapse a run of punctuation to a single mark: "!!!" -> "!".
    text = re.sub(r"[،.؟!][\s،.؟!]*", lambda m: m.group(0).lstrip()[0] + " ", text)
    text = re.sub(r"\s+([،.؟!])", r"\1", text)
    text = re.sub(r"([،.؟!])(?=\S)", r"\1 ", text)
    text = RE_WS.sub(" ", text).strip()

    # Punctuation with no words left is not speakable text: return nothing
    # rather than handing the model a line of bare marks.
    if not any("\u0621" <= ch <= "\u064a" for ch in text):
        return ""
    return text.strip(" \u060c.\u061f!").strip()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

NUMBER_CASES: List[Tuple[int, str]] = [
    (0, "صفر"),
    (1, "واحد"),
    (2, "اتنين"),
    (7, "سبعة"),
    (10, "عشرة"),
    (11, "حداشر"),
    (15, "خمستاشر"),
    (19, "تسعتاشر"),
    (20, "عشرين"),
    (21, "واحد و عشرين"),
    (24, "اربعة و عشرين"),
    (76, "ستة و سبعين"),
    (99, "تسعة و تسعين"),
    (100, "مية"),
    (200, "متين"),
    (300, "تلتمية"),
    (800, "تمنمية"),
    # The rule from the brief: no waw between the hundred and the remainder.
    (876, "تمنمية ستة و سبعين"),
    (101, "مية واحد"),
    (999, "تسعمية تسعة و تسعين"),
    (1000, "الف"),
    (2000, "الفين"),
    (3000, "تلاتة الاف"),
    (10000, "عشرة الاف"),
    (11000, "حداشر الف"),
    # The headline case: 2024 must not contain a waw after "الفين".
    (2024, "الفين اربعة و عشرين"),
    (1986, "الف تسعمية ستة و تمانين"),
    (1000000, "مليون"),
    (2000000, "مليونين"),
    (5000000, "خمسة ملايين"),
    (1500000, "مليون خمسمية الف"),
]

TEXT_CASES: List[Tuple[str, Sequence[str], Sequence[str]]] = [
    # (input, must contain, must NOT contain)
    ("انا اتولدت سنة 1986", ["الف تسعمية ستة و تمانين"], ["1986", "الف و"]),
    ("السنة دي 2024", ["الفين اربعة و عشرين"], ["الفين و اربعة"]),
    ("المبلغ 876 جنيه", ["تمنمية ستة و سبعين", "جنيه"], ["تمنمية و ستة"]),
    ("السعر 250 ج.م", ["متين خمسين", "جنيه"], ["250"]),
    ("الخصم 25%", ["خمسة و عشرين", "في المية"], ["%", "25"]),
    ("الساعة 3:30", ["التالتة", "نص"], [":"]),
    ("الساعة 7:45", ["التامنة", "الا ربع"], []),
    ("الميعاد 12/5/2024", ["مايو", "الفين اربعة و عشرين"], ["/"]),
    ("ابعتلي على ahmed@gmail.com", ["ات", "جيميل", "نقطة", "كوم"], ["@"]),
    ("شوف www.youtube.com", ["يوتيوب", "نقطة", "كوم"], ["www"]),
    ("الدكتور د. احمد جه", ["دكتور"], []),
    ("رقمي 01012345678", ["صفر", "واحد"], ["01012345678"]),
    ("جااااااامد اوي", ["جاامد"], ["جااااااامد"]),
    ("عندي 3.5 كيلو", ["تلاتة", "فاصلة", "خمسة"], ["3.5"]),
    ("من 10 - 20 يوم", ["من", "عشرة", "عشرين"], ["-"]),
    ("خد 1/2 كيلو", ["نص"], ["1/2"]),
    ("الترتيب ال5", ["الخامس"], []),
    ("درجة الحرارة 30°", ["تلاتين", "درجة"], ["°"]),
    ("عَلَم مِصر بيرفرف", ["علم", "مصر"], ["َ", "ِ"]),
    ("١٩٨٦ سنة حلوة", ["الف تسعمية ستة و تمانين"], ["١٩٨٦"]),
]


def _run_self_test() -> int:
    passed = failed = 0

    print("=" * 74)
    print("EGYPTIAN ARABIC NUMBER READING")
    print("=" * 74)
    print("Rule: the linking waw appears ONLY between a unit and a ten.")
    print("      Magnitude groups are juxtaposed with no waw.")
    print()
    for n, expected in NUMBER_CASES:
        got = number_to_words(n)
        ok = got == expected
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
        mark = "ok  " if ok else "FAIL"
        print(f"  {mark} {n:>9}  ->  {got}")
        if not ok:
            print(f"       {'':>9}      expected: {expected}")

    print()
    print("=" * 74)
    print("FULL TEXT NORMALIZATION")
    print("=" * 74)
    for src, must, must_not in TEXT_CASES:
        got = normalize_egyptian(src)
        missing = [m for m in must if m not in got]
        leaked = [m for m in must_not if m in got]
        ok = not missing and not leaked
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
        print(f"  {'ok  ' if ok else 'FAIL'} {src}")
        print(f"       -> {got}")
        if missing:
            print(f"       MISSING: {missing}")
        if leaked:
            print(f"       LEAKED:  {leaked}")

    print()
    print("=" * 74)
    print("HARD CASES (mixed content in one line)")
    print("=" * 74)
    hard = [
        "الاجتماع يوم 15/3/2024 الساعة 10:30 ص في قاعة 5",
        "اتصل على 01098765432 او ابعت لـ info@company.com.eg",
        "الشركة حققت ارباح 2,500,000 جنيه بزيادة 15% عن 2023",
        "د. محمد قال ان 3/4 من الطلاب نجحوا بنسبة 75%",
        "العنوان 25 شارع التحرير الدور ال3 شقة 12 القاهرة",
        "حمل التطبيق من www.google.com/play وقيمه 5 نجوم",
        "السعر $50 يعني حوالي 2400 ج.م تقريبا",
        "من سنة 1986 لسنة 2024 يعني 38 سنة",
    ]
    for h in hard:
        out = normalize_egyptian(h)
        leftover = [c for c in out if c.isdigit() or c.isascii() and c.isalpha()]
        clean = not leftover
        passed, failed = (passed + 1, failed) if clean else (passed, failed + 1)
        print(f"  {'ok  ' if clean else 'FAIL'} {h}")
        print(f"       -> {out}")
        if leftover:
            print(f"       LEFTOVER unspeakable characters: {sorted(set(leftover))}")

    print()
    print("=" * 74)
    print(f"RESULT: {passed} passed, {failed} failed")
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(_run_self_test())
