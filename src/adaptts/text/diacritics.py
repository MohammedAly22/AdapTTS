"""Pronunciation labels from diacritized text.

## Why this replaces unsupervised discovery

The original design clustered acoustic embeddings of each word occurrence and
called the clusters "pronunciations". On 68 hours of real Egyptian narration
that failed, and it failed in a way that looked like success: the top discovered
words were الجرس, التعليقات, لايك, الوصف, الرابط, البلاي. Those are the channel's
subscribe pitch. They appear in exactly two acoustic contexts, the scripted
promo read in a fixed fast cadence and ordinary narration, and the clustering
found that split. Meanwhile علم, مصر and دول each got a single code.

The reason is structural. A mean-pooled self-supervised embedding over a word
span encodes speaking rate, energy, pitch and channel far more strongly than
vowel identity. Stability, silhouette and separation gates filter *noise*; a
confound that is reproducible and well separated passes all of them.

Diacritics are a direct observation of the vowels instead of a proxy. A
diacritizer that writes عَلَم in one sentence and عِلْم in another has stated the
distinction we need, and it says nothing about الجرس because there is nothing to
say.

## What this module does and does not assume

It takes *already diacritized* text and derives, per word type, the set of
distinct vowel patterns it takes across the corpus. It does not run the
diacritizer; that lives in a separate script so this module stays dependency
free and testable.

Crucially, the diacritizer is **not treated as ground truth**. It is noisy, so:

* A pattern must occur ``min_pattern_count`` times before it counts as a
  reading. A single odd diacritization is an error, not a homograph.
* A pattern must hold at least ``min_pattern_frac`` of the word's occurrences.
* Patterns that differ only in case-ending (the final short vowel, which Egyptian
  speech largely drops) are merged, because they are the same word.

The shipped model never sees diacritics. They exist only to label the training
data, exactly as a forced aligner exists only to find spans.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Character classes
# ---------------------------------------------------------------------------

FATHA, DAMMA, KASRA = "َ", "ُ", "ِ"
FATHATAN, DAMMATAN, KASRATAN = "ً", "ٌ", "ٍ"
SUKUN, SHADDA = "ْ", "ّ"
SUPERSCRIPT_ALEF = "ٰ"
TATWEEL = "ـ"

SHORT_VOWELS = frozenset({FATHA, DAMMA, KASRA})
TANWEEN = frozenset({FATHATAN, DAMMATAN, KASRATAN})
DIACRITICS = frozenset(
    {FATHA, DAMMA, KASRA, FATHATAN, DAMMATAN, KASRATAN, SUKUN, SHADDA, SUPERSCRIPT_ALEF}
    | {chr(c) for c in range(0x064B, 0x0653)}
)

# ---------------------------------------------------------------------------
# Phoneme variants
# ---------------------------------------------------------------------------
#
# CATT emits two very different kinds of non-standard character, and treating
# them alike destroys information. Measured on real ECA output (see
# PHONOLOGY.md):
#
#   ؞ (U+061E) attaches to exactly ج, ق and ف, and marks a NON-DEFAULT
#     realisation of that letter. It is phonemic and must be preserved.
#
#   ^ < > ` … attach to anything (measured on و ع م ر ز ن ه د) with no
#     phonological pattern. They are positional noise and are deleted.
#
# The triple dot is rewritten to '~' because U+061E is invisible in most
# terminals and is silently lost in copy-paste, and because a user typing an
# override needs a character they can actually enter.

VARIANT_MARK = "~"
_TRIPLE_DOT = "؞"

# Which letters the variant mark may attach to, and what it means. The value is
# for documentation and error messages; the model learns the acoustics itself.
VARIANT_LETTERS = {
    "ق": "qaf -> hamza (glottal stop)",   # ق
    "ج": "geem /g/ -> /zh/ (jeem)",       # ج
    "ف": "faa /f/ -> /v/",                # ف
}

# Sukun on a word-final taa-marbuta means the /t/ surfaces: مَدِينَة is /madiina/
# but مَدِينَةْ is /madiinat/. That is a real contrast, unlike the case endings
# around it, so it is carried as a variant rather than suppressed.
TAA_MARBUTA = "ة"

# Noise, with no phonological content. Deleted outright.
KNOWN_JUNK = frozenset({"^", ">", "<", "…", "`"})


def is_arabic_letter(ch: str) -> bool:
    return "ء" <= ch <= "ي" or ch in "ٱپچژڤگ"


def normalize_variant_marks(text: str) -> str:
    """Rewrite CATT's triple dot as ``~`` and drop the noise characters.

    Run this on diacritizer output before anything else looks at it. The
    distinction it preserves is the whole point: ``؞`` carries pronunciation,
    ``^`` does not.

    A variant mark that did not land on ق, ج or ف is dropped, because the
    measurement says those are the only letters it applies to and anything else
    is a stray.
    """
    out: List[str] = []
    for ch in text:
        if ch in KNOWN_JUNK:
            continue
        if ch in (_TRIPLE_DOT, VARIANT_MARK):
            # Attaches to the last letter emitted, so check that letter.
            prev_letter = next(
                (c for c in reversed(out) if is_arabic_letter(c)), ""
            )
            if prev_letter in VARIANT_LETTERS:
                out.append(VARIANT_MARK)
            continue
        out.append(ch)
    return "".join(out)


def strip_junk(text: str) -> str:
    """Remove every non-linguistic character, variant marks included.

    Used where only the vowel pattern matters, since a variant mark must never
    create a reading: CATT marks the qaf of قرار inconsistently between
    contexts, so promoting the mark to a pattern distinction would split that
    word on a diacritizer error. Use :func:`normalize_variant_marks` instead
    wherever the pronunciation detail has to survive.
    """
    return "".join(
        ch for ch in text
        if ch not in KNOWN_JUNK
        and ch not in (_TRIPLE_DOT, VARIANT_MARK)
        and (ch.isspace() or is_arabic_letter(ch) or ch in DIACRITICS)
    )


def strip_diacritics(word: str) -> str:
    """The bare consonant skeleton, which is what the model actually reads."""
    return "".join(
        ch for ch in word
        if ch not in DIACRITICS
        and ch != TATWEEL
        and ch not in KNOWN_JUNK
        and ch not in (_TRIPLE_DOT, VARIANT_MARK)
    )


# Variant feature indices. 0 means "default realisation", which is the common
# case, so an all-zero vector costs nothing and means "nothing unusual here".
VARIANT_NONE = 0
VARIANT_LETTER = 1      # ق/ج/ف realised non-natively (the triple dot)
VARIANT_TAA_T = 2       # word-final ة pronounced /t/ (sukun on taa-marbuta)
N_VARIANTS = 3


def letter_variants(word: str) -> List[int]:
    """Per-letter phoneme-variant codes for one diacritized word.

    Returns one code per letter of ``strip_diacritics(word)``, so the result
    lines up with the characters the model actually sees. This is the channel
    through which pronunciation detail reaches the acoustic model without
    entering the vowel pattern, which must stay free of it.

    Two sources, both measured rather than assumed:

    * a variant mark (``~``, from CATT's ``؞``) on ق, ج or ف;
    * sukun on a word-final ة, which makes the /t/ surface.
    """
    letters: List[str] = []
    variants: List[int] = []
    pending_marks: List[str] = []

    for ch in word:
        if ch == TATWEEL or ch in KNOWN_JUNK:
            continue
        if ch in (_TRIPLE_DOT, VARIANT_MARK):
            if letters and letters[-1] in VARIANT_LETTERS:
                variants[-1] = VARIANT_LETTER
            continue
        if ch in DIACRITICS:
            if letters:
                pending_marks.append(ch)
                # Sukun on a final taa-marbuta is resolved after the loop, when
                # "final" is actually known.
            continue
        if not is_arabic_letter(ch):
            continue
        letters.append(ch)
        variants.append(VARIANT_NONE)
        pending_marks = []

    # The /t/ contrast, only meaningful on the last letter.
    if letters and letters[-1] == TAA_MARBUTA:
        marks_on_last = _marks_on_final_letter(word)
        if SUKUN in marks_on_last:
            variants[-1] = VARIANT_TAA_T

    return variants


def _marks_on_final_letter(word: str) -> str:
    """The diacritics attached to the last letter of ``word``.

    Marks follow their letter, so scanning backwards these come *before* the
    final letter is reached; everything collected up to that point belongs to
    it.
    """
    marks: List[str] = []
    for ch in reversed(word):
        if ch in DIACRITICS:
            marks.append(ch)
            continue
        if ch in (_TRIPLE_DOT, VARIANT_MARK) or ch in KNOWN_JUNK or ch == TATWEEL:
            continue
        if is_arabic_letter(ch):
            break          # reached the final letter; marks after it are its own
    return "".join(reversed(marks))


# Hamza and alef forms a diacritizer may add or change. Folding these lets an
# output token match its input word: CATT writes أنَاْ for انا and إِلدِّوَلْ for
# الدول, and those are the same word.
_ALEF_FOLD = {
    "أ": "ا",  # أ
    "إ": "ا",  # إ
    "آ": "ا",  # آ
    "ٱ": "ا",  # ٱ
    "ى": "ي",  # ى -> ي
    "ة": "ه",  # ة -> ه
}


def fold_orthography(word: str) -> str:
    """Normalize hamza, alef-maqsura and taa-marbuta for matching only.

    Used to decide whether a diacritized token corresponds to a given input
    word. Never used to change the text the model sees.
    """
    return "".join(_ALEF_FOLD.get(ch, ch) for ch in word)


def match_key(word: str) -> str:
    """Key under which an input word and its diacritized form should agree."""
    return fold_orthography(strip_diacritics(word))


def word_identity(word: str) -> str:
    """The lexicon key for a word, folding spellings that are one word.

    The corpus writes the feminine ending both ways: مدينة and مدينه are the
    same word, and so are الحكاية and الحكايه. Keyed by their raw spelling
    they become two entries with two code sets, which splits the training data
    for one word and, because CATT diacritizes the ة spelling well and the ه
    spelling badly, lets the bad half define its own readings.

    Alef-maqsura and hamza forms fold for the same reason. This is identity
    only: the text the model reads is never rewritten.
    """
    return fold_orthography(strip_diacritics(word))


# ---------------------------------------------------------------------------
# Vowel patterns
# ---------------------------------------------------------------------------


def letter_marks(word: str) -> List[Tuple[str, str]]:
    """Split a diacritized word into ``(letter, marks)`` pairs.

    Diacritics bind to the letter they follow, so any comparison of readings has
    to respect that grouping. Working on a flat list of marks silently confuses
    an interior vowel with a case ending.

    Junk is dropped here rather than by the caller. CATT emits its stray
    characters *inside* words, between a letter and its mark: real output for
    مصر is ``م^َصْرِ``. Treating ``^`` as a letter would give it the following
    fatha and add a phantom slot, so the same word would take two different
    patterns and be counted as a homograph. Everything that is not an Arabic
    letter or a diacritic is therefore skipped.
    """
    out: List[Tuple[str, str]] = []
    for ch in word:
        if ch == TATWEEL:
            continue
        if ch in DIACRITICS:
            if out:
                out[-1] = (out[-1][0], out[-1][1] + ch)
            continue
        if not is_arabic_letter(ch):
            continue
        out.append((ch, ""))
    return out


LONG_VOWELS = frozenset({"\u0627", "\u0648", "\u064a", "\u0649"})  # ا و ي ى

# Which short vowel each long vowel lengthens. A sukun on the long vowel after
# its matching short vowel is spurious: the letter IS the vowel.
_LENGTHENS = {"\u0627": FATHA, "\u0648": DAMMA, "\u064a": KASRA, "\u0649": FATHA}


def clean_marks(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Remove diacritizer artifacts that cannot encode a pronunciation.

    Measured on real CATT output, these three cases account for the large
    majority of spurious "readings":

    * a mark on the word-initial letter (Arabic words do not start vowelless);
    * a sukun on a long vowel that follows its matching short vowel (the letter
      is the vowel, so it cannot also be vowelless);
    * marks inside the definite article ال, which is invariant.

    Each is a phonological impossibility rather than a stylistic choice, so
    removing them cannot erase a real distinction.
    """
    if not pairs:
        return pairs
    out = [(ltr, mk) for ltr, mk in pairs]

    # 1. Word-initial mark.
    out[0] = (out[0][0], "")

    # 2. Sukun on a long vowel letter. Alef, waw and ya in a non-initial
    #    position are the vowel itself, so "no vowel here" is meaningless on
    #    them. CATT emits this inconsistently: قَالْ and قَاْلْ are one word.
    for i in range(1, len(out)):
        ltr, mk = out[i]
        if SUKUN in mk and ltr in LONG_VOWELS:
            out[i] = (ltr, mk.replace(SUKUN, ""))

    # 3. The definite article carries no contrast.
    if len(out) >= 2 and out[0][0] == "\u0627" and out[1][0] == "\u0644":
        out[1] = (out[1][0], "")

    # 4. Taa marbuta and word-final heh take no vowel of their own, and the mark
    #    on the letter before is a case ending. CATT flips these freely:
    #    وَصْفَه vs وَصْفُه, عَمَلُه vs عَمَلَه are the same word each time,
    #    and Egyptian speech drops case endings, so none of it is contrastive.
    #
    #    The corpus spells this ending both ways (مدينة / مدينه) for the same
    #    word, so both letters are handled here and word identity folds them
    #    together upstream. The one real contrast here, sukun on a final ة
    #    meaning the /t/ surfaces (مَدِينَةْ = /madiinat/), is carried by
    #    letter_variants as a phoneme feature rather than by the vowel pattern,
    #    so clearing it here loses nothing and keeps the pattern free of it.
    if out and out[-1][0] in (TAA_MARBUTA, "ه"):
        out[-1] = (out[-1][0], "")
        if len(out) >= 2:
            out[-2] = (out[-2][0], "")
    # 5. Shadda on the final cluster. Gemination is phonemic in general, which
    #    is why it survives elsewhere, but CATT is inconsistent about it at the
    #    very end of a word (مِية vs مِيَّة) where Egyptian does not contrast it.
    if len(out) >= 2 and SHADDA in out[-1][1]:
        out[-1] = (out[-1][0], out[-1][1].replace(SHADDA, ""))

    return out


def vowel_pattern(word: str, drop_case_ending: bool = True) -> str:
    """A comparable signature of how a word is voiced.

    Two occurrences of the same spelling share a pattern when they are
    pronounced the same way. The signature records, per letter, the marks on it,
    so عَلَم and عِلْم differ while two spellings of the same reading agree.

    ``drop_case_ending`` ignores the mark on the **final letter only**.
    Egyptian speech largely drops case endings, so مَصْرِ and مَصْرْ are one word
    said one way; keeping them apart would manufacture homographs from every
    noun in the corpus.

    Sukun is kept in interior positions: it is the difference between عَلَم
    (fatha on the lam) and عِلْم (sukun on the lam), which is exactly the
    distinction this whole system exists to capture.
    """
    pairs = clean_marks(letter_marks(word))
    if not pairs:
        return ""

    if drop_case_ending and len(pairs) > 1:
        # Drop the final VOWEL only. Shadda on the last letter is consonant
        # doubling, not a case ending: مُصِرّ really ends in a doubled ر, and
        # clearing it would merge مُصِرّ with مُصِر.
        last_letter, last_marks = pairs[-1]
        kept = SHADDA if SHADDA in last_marks else ""
        pairs = pairs[:-1] + [(last_letter, kept)]

    # Shadda is doubling, which is phonemic (مُصِرّ), so it is preserved. The
    # order of marks on one letter is normalized so shadda+vowel and
    # vowel+shadda compare equal.
    parts = []
    for i, (_, marks) in enumerate(pairs):
        if not marks:
            parts.append("-")
            continue
        norm = ("+" if SHADDA in marks else "") + "".join(
            sorted(m for m in marks if m != SHADDA)
        )
        parts.append(norm or "-")
    return "|".join(parts)


def _pattern_slots(pairs: List[Tuple[str, str]]) -> List[str]:
    """The per-letter slot strings a vowel pattern is built from."""
    out = []
    for _, marks in pairs:
        norm = ("+" if SHADDA in marks else "") + "".join(
            sorted(m for m in marks if m != SHADDA)
        )
        out.append(norm or "-")
    return out


@dataclass
class WordReadings:
    """The distinct readings one word type takes across the corpus."""

    word: str
    patterns: List[str]          # index == pronunciation code
    counts: List[int]
    examples: List[str]          # one diacritized form per pattern
    total: int

    @property
    def n_codes(self) -> int:
        return len(self.patterns)

    def code_of(self, diacritized: str) -> int:
        """Which code does this occurrence belong to? -1 when unknown."""
        p = vowel_pattern(diacritized)
        try:
            return self.patterns.index(p)
        except ValueError:
            return -1

    def to_json(self) -> dict:
        return {
            "word": self.word,
            "patterns": self.patterns,
            "counts": self.counts,
            "examples": self.examples,
            "total": self.total,
        }

    @classmethod
    def from_json(cls, d: dict) -> "WordReadings":
        return cls(
            word=d["word"],
            patterns=list(d["patterns"]),
            counts=list(d["counts"]),
            examples=list(d["examples"]),
            total=int(d["total"]),
        )


def align_tokens(
    plain: Sequence[str], diacritized: Sequence[str]
) -> List[Optional[str]]:
    """Pair each plain word with its diacritized form.

    A diacritizer is expected to preserve token count, but it can merge or drop
    a token. Rather than trusting position alone, each pair is verified by
    comparing folded skeletons, and a mismatch yields ``None`` so that
    occurrence is skipped instead of mislabelled.
    """
    out: List[Optional[str]] = []
    if len(plain) == len(diacritized):
        for p, d in zip(plain, diacritized):
            out.append(d if match_key(d) == match_key(p) else None)
        return out

    # Token counts differ: walk both sequences and match greedily.
    di = 0
    for p in plain:
        want = match_key(p)
        found = None
        for probe in range(di, min(di + 3, len(diacritized))):
            if match_key(diacritized[probe]) == want:
                found = diacritized[probe]
                di = probe + 1
                break
        out.append(found)
    return out


def _context_agreement(
    by_context: Dict[Tuple[str, str], Counter]
) -> Tuple[float, int]:
    """How consistently does an identical context give the same pattern?

    Returns ``(agreement, n_occurrences_in_repeated_contexts)``. Contexts seen
    only once carry no evidence either way and are ignored.

    This is the measurement that separates homography from diacritizer
    uncertainty. A word whose reading follows its neighbours is a homograph; one
    whose identical neighbours produce different readings is noise.
    """
    repeated = [c for c in by_context.values() if sum(c.values()) > 1]
    if not repeated:
        return 1.0, 0
    agree = sum(max(c.values()) for c in repeated)
    total = sum(sum(c.values()) for c in repeated)
    return (agree / total if total else 1.0), total


def collect_readings(
    sentences: Sequence[Tuple[Sequence[str], Sequence[str]]],
    *,
    min_word_freq: int = 5,
    min_pattern_count: int = 3,
    min_pattern_frac: float = 0.08,
    max_codes: int = 4,
    max_raw_patterns: int = 4,
    min_context_agreement: float = 0.75,
) -> Dict[str, WordReadings]:
    """Derive per-word readings from aligned (plain, diacritized) sentences.

    The thresholds exist because the diacritizer is imperfect. A pattern seen
    once is an error; a pattern seen many times across many sentences is a
    reading. Requiring both an absolute count and a share of the word's
    occurrences rejects both rare noise and a systematic minority slip.

    Two further gates separate real homography from diacritizer uncertainty,
    which counts alone cannot distinguish:

    * ``max_raw_patterns``: a word the diacritizer gives five or more different
      patterns is one it is unsure about, not one with five readings.
    * ``min_context_agreement``: a genuine reading is determined by context, so
      where the same neighbouring words repeat, the pattern must agree. Measured
      on real data, real homographs score 80-100% here (مصر and عالم both 100%,
      دول 93%) while noise scores 50% or has no repeated context at all.

    Without these, the labels are mostly unlearnable: a context encoder trained
    on the unfiltered set scored 67.9% held out against a 68.3% majority
    baseline, meaning it learned nothing beyond the prior.
    """
    seen: Dict[str, Counter] = defaultdict(Counter)
    example: Dict[Tuple[str, str], str] = {}
    # (word, left, right) -> patterns observed in that exact context
    contexts: Dict[str, Dict[Tuple[str, str], Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )

    for plain, diac in sentences:
        aligned = align_tokens(plain, diac)
        for i, (p, d) in enumerate(zip(plain, aligned)):
            if d is None:
                continue
            pat = vowel_pattern(d)
            if not pat:
                continue  # no diacritics at all: nothing to learn
            # Key by folded identity so the two spellings of a feminine ending
            # are one word rather than two half-populated entries.
            key = word_identity(p)
            seen[key][pat] += 1
            example.setdefault((key, pat), d)
            left = word_identity(plain[i - 1]) if i > 0 else "<s>"
            right = word_identity(plain[i + 1]) if i + 1 < len(plain) else "</s>"
            contexts[key][(left, right)][pat] += 1

    readings: Dict[str, WordReadings] = {}
    for word, counter in seen.items():
        total = sum(counter.values())
        if total < min_word_freq:
            continue

        # Gate 1: too many raw patterns means the diacritizer is guessing.
        if len(counter) > max_raw_patterns:
            continue

        kept = [
            (pat, n) for pat, n in counter.most_common()
            if n >= min_pattern_count and n / total >= min_pattern_frac
        ][:max_codes]
        if not kept:
            continue

        # Gate 2: where a context repeats, the reading must agree. A word whose
        # identical surroundings yield different patterns is not a homograph.
        if len(kept) > 1:
            agree, n_rep = _context_agreement(contexts[word])
            if n_rep >= 2 and agree < min_context_agreement:
                continue
        readings[word] = WordReadings(
            word=word,
            patterns=[p for p, _ in kept],
            counts=[n for _, n in kept],
            examples=[example[(word, p)] for p, _ in kept],
            total=total,
        )
    return readings


class ReadingLexicon:
    """Word type -> its distinct readings, derived from diacritized text.

    Despite the name this is not a handwritten pronunciation dictionary. It is
    computed from the corpus and regenerated whenever the corpus changes.
    """

    def __init__(self, entries: Dict[str, WordReadings], max_codes: int) -> None:
        self.entries = entries
        self.max_codes = max_codes

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, word: str) -> Optional["WordReadings"]:
        """The entry for a word, folding spelling variants onto one key.

        Entries are keyed by :func:`word_identity`, so a caller holding the raw
        spelling (مدينة where the entry is under مدينه) still finds it. The raw
        key is tried first so a lexicon built before folding still loads.
        """
        e = self.entries.get(word)
        if e is not None:
            return e
        return self.entries.get(word_identity(word))

    def n_codes(self, word: str) -> int:
        e = self.get(word)
        return e.n_codes if e is not None else 1

    def is_ambiguous(self, word: str) -> bool:
        return self.n_codes(word) > 1

    @property
    def ambiguous_words(self) -> List[str]:
        return sorted(w for w, e in self.entries.items() if e.n_codes > 1)

    def code_of(self, word: str, diacritized: str) -> int:
        e = self.get(word)
        return e.code_of(diacritized) if e is not None else 0

    def code_from_partial_marks(self, word: str) -> Tuple[int, str]:
        """Resolve a code from however many diacritics the user actually typed.

        This is the override channel. Asking someone to pass a code integer
        requires them to know that code 1 means عِلْم, which is unknowable: codes
        are assigned by corpus frequency and change when the lexicon is rebuilt.
        Writing the vowel is the interface people already have.

        ``word`` is one token carrying some marks, e.g. مُصِرّ or just مُصِر. A
        full diacritization is not required: whatever marks are present are
        compared against each known reading, and positions left bare count as
        "no opinion" rather than as an assertion of vowellessness.

        The user's word is reduced with :func:`vowel_pattern`, the same function
        that produced every stored pattern, so the two are always in the same
        space. That reduction deliberately discards the word-initial position
        (where diacritizers emit spurious marks) and the case ending (which
        Egyptian drops), so a mark typed on either is uninformative here. It is
        not wrong, just not something the lexicon distinguishes, so it is
        skipped rather than counted as a conflict.

        Returns ``(code, reason)``. A code of -1 means unresolved, with the
        reason, so a caller reports it rather than silently mispronouncing.
        """
        e = self.get(word)
        if e is None:
            return -1, "word not in the lexicon"
        if e.n_codes == 1:
            return 0, "single reading"

        raw = letter_marks(word)
        if not any(mk for _, mk in raw):
            return -1, "no diacritics supplied"

        # Same reduction as every stored pattern, so slots are comparable.
        user_slots = vowel_pattern(word).split("|")
        supplied = [bool(mk) for _, mk in raw]
        if len(supplied) != len(user_slots):
            return -1, "could not parse the marks given"

        scored: List[Tuple[int, int, int]] = []     # (agree, conflict, code)
        for code, pat in enumerate(e.patterns):
            slots = pat.split("|")
            if len(slots) != len(user_slots):
                continue
            agree = conflict = 0
            for i, (us, theirs) in enumerate(zip(user_slots, slots)):
                if not supplied[i]:
                    continue            # user said nothing about this letter
                if us == "-" and theirs == "-":
                    # A position the reduction discards on both sides: the mark
                    # was given but carries no distinction here.
                    continue
                if us == theirs:
                    agree += 1
                else:
                    conflict += 1
            scored.append((agree, conflict, code))

        if not scored:
            return -1, "marks do not line up with any known reading"

        # A conflict means the user stated a vowel that reading does not have.
        clean = [s for s in scored if s[1] == 0 and s[0] > 0]
        if not clean:
            if not any(s[0] or s[1] for s in scored):
                return -1, (
                    "the marks given fall on positions the lexicon does not "
                    "distinguish (word-initial letter or case ending); mark an "
                    "interior vowel instead"
                )
            return -1, "marks conflict with every known reading"
        clean.sort(key=lambda s: -s[0])
        best = clean[0]
        if len([s for s in clean if s[0] == best[0]]) > 1:
            return -1, "marks match several readings equally"
        return best[2], f"matched {best[0]} mark(s) with no conflict"

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_codes": self.max_codes,
            "entries": [e.to_json() for e in self.entries.values()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: Path) -> "ReadingLexicon":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        entries = {}
        for d in payload["entries"]:
            wr = WordReadings.from_json(d)
            entries[wr.word] = wr
        return cls(entries, int(payload["max_codes"]))


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("strip_junk:")
    dirty = "عَلَمْ^ مَصْر؞ْ >دول"
    print(f"  {dirty!r}\n  -> {strip_junk(dirty)!r}")

    print("\nvowel_pattern (case ending dropped):")
    for w in ["عَلَمْ", "عِلْمْ", "مَصْرِ", "مُصِرِّ", "الْجَرَسْ"]:
        print(f"  {w:<10} bare={strip_diacritics(w):<8} pattern={vowel_pattern(w)!r}")

    print("\nmatch_key folds hamza so a diacritized token matches its input:")
    for a, b in [("انا", "أنَاْ"), ("الدول", "إِلدِّوَلْ"), ("السلام", "ألسَّلَامُ")]:
        ok = match_key(a) == match_key(b)
        print(f"  {a:<8} vs {b:<12} -> {ok}")

    print("\ncollect_readings on a tiny corpus:")
    corpus = [
        (["انا", "شوفت", "علم", "مصر"], ["أنَاْ", "شُوْفْتِ", "عَلَمْ", "مَصْرِ"]),
        (["علم", "الفيزيا", "مهم"], ["عِلْمْ", "الْفِيْزْيَا", "مُهِمّْ"]),
        (["علم", "النفس", "حلو"], ["عِلْمْ", "النَّفْسْ", "حِلْوِ"]),
        (["شوفت", "علم", "كبير"], ["شُوْفْتِ", "عَلَمْ", "كِبِيرْ"]),
        (["علم", "الاحياء"], ["عِلْمْ", "الاحْيَاءْ"]),
        (["رفعوا", "علم", "النادي"], ["رَفَعُوا", "عَلَمْ", "النَّادِي"]),
    ]
    r = collect_readings(corpus, min_word_freq=3, min_pattern_count=2, min_pattern_frac=0.2)
    for w, e in sorted(r.items()):
        mark = "  <-- ambiguous" if e.n_codes > 1 else ""
        print(f"  {w:<10} {e.n_codes} reading(s) {e.counts} {e.examples}{mark}")
