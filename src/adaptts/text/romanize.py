"""Arabic to Latin transliteration, for CTC forced alignment only.

The MMS forced aligner (``MahmoudAshraf/mms-300m-1130-forced-aligner``) is a
massively multilingual CTC model whose output vocabulary is **romanized**: 31
tokens covering the Latin letters plus an apostrophe. It does not emit Arabic
script. To align Arabic audio with it, the transcript has to be romanized in
the same way its training data was, which is what this module does.

Two things to be clear about:

1. **This is not a pronunciation model.** It is a script mapping used only to
   drive alignment. It never decides how a homograph is read; that remains the
   job of the discovered pronunciation codes. A rough romanization is fine
   here, because we only need frame boundaries, not phonetics.

2. **Character counts are tracked.** One Arabic character can romanize to two
   Latin ones (``ش`` -> ``sh``). The aligner needs to map Latin frames back to
   Arabic character positions, so :func:`romanize_with_map` returns, for every
   emitted Latin character, the index of the Arabic character it came from.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Mapping tuned to the aligner's 31-token vocabulary:
# a i e n o u t s r m k l d g h y b p w c v j z f ' q x
ARABIC_TO_LATIN: Dict[str, str] = {
    # Alef and its hamza forms.
    "ا": "a",   # ا
    "أ": "a",   # أ
    "إ": "i",   # إ
    "آ": "a",   # آ
    "ء": "'",   # ء
    "ئ": "'",   # ئ
    "ؤ": "'",   # ؤ
    "ى": "a",   # ى  alef maqsura
    # Consonants.
    "ب": "b",   # ب
    "ت": "t",   # ت
    "ث": "s",   # ث  (Egyptian merges to /s/)
    "ج": "g",   # ج  (Egyptian /g/, not /dʒ/)
    "ح": "h",   # ح
    "خ": "kh",  # خ
    "د": "d",   # د
    "ذ": "z",   # ذ  (Egyptian merges to /z/)
    "ر": "r",   # ر
    "ز": "z",   # ز
    "س": "s",   # س
    "ش": "sh",  # ش
    "ص": "s",   # ص
    "ض": "d",   # ض
    "ط": "t",   # ط
    "ظ": "z",   # ظ
    "ع": "'",   # ع
    "غ": "gh",  # غ
    "ف": "f",   # ف
    "ق": "'",   # ق  (Egyptian glottal stop)
    "ك": "k",   # ك
    "ل": "l",   # ل
    "م": "m",   # م
    "ن": "n",   # ن
    "ه": "h",   # ه
    "و": "w",   # و
    "ي": "y",   # ي
    "ة": "a",   # ة  taa marbuta
    # Persian/Urdu letters that appear in loanwords.
    "پ": "p",   # پ
    "چ": "ch",  # چ
    "ژ": "j",   # ژ
    "ڤ": "v",   # ڤ
    "گ": "g",   # گ
}

# Short vowels, if a diacritized transcript ever reaches us.
DIACRITIC_TO_LATIN: Dict[str, str] = {
    "َ": "a",  # fatha
    "ُ": "u",  # damma
    "ِ": "i",  # kasra
    "ً": "an",
    "ٌ": "un",
    "ٍ": "in",
}


def romanize_with_map(text: str) -> Tuple[str, List[int]]:
    """Romanize ``text`` and record where each Latin character came from.

    Returns ``(latin, source_index)`` where ``source_index[i]`` is the index in
    ``text`` of the Arabic character that produced ``latin[i]``. That mapping is
    what lets the aligner convert Latin-level frame spans back into Arabic
    character positions, and from there into word spans.

    Spaces are preserved one-for-one so word boundaries survive.
    """
    out: List[str] = []
    src: List[int] = []

    for i, ch in enumerate(text):
        if ch.isspace():
            out.append(" ")
            src.append(i)
            continue
        if ch in DIACRITIC_TO_LATIN:
            for c in DIACRITIC_TO_LATIN[ch]:
                out.append(c)
                src.append(i)
            continue

        mapped = ARABIC_TO_LATIN.get(ch)
        if mapped is None:
            if ch.isascii() and ch.isalpha():
                mapped = ch.lower()  # already Latin
            elif ch.isdigit():
                continue  # normalization should have removed these
            else:
                continue  # punctuation and anything unmappable
        for c in mapped:
            out.append(c)
            src.append(i)

    return "".join(out), src


def romanize(text: str) -> str:
    """Romanize without the index map."""
    return romanize_with_map(text)[0]


def build_ctc_targets(
    text: str,
    vocab: Dict[str, int],
    unk_id: int,
) -> Tuple[List[int], List[int]]:
    """Convert Arabic text into CTC target ids plus their source indices.

    Returns ``(ids, source_index)``, both the same length. Characters the
    aligner's vocabulary cannot represent are dropped rather than mapped to
    ``<unk>``, because an unknown token contributes no useful alignment
    evidence and only blurs the spans around it.
    """
    latin, src = romanize_with_map(text)
    ids: List[int] = []
    keep: List[int] = []
    for ch, s in zip(latin, src):
        if ch == " ":
            # The aligner has no word-delimiter token, so spaces are simply not
            # emitted. Word boundaries are recovered from the source indices.
            continue
        tid = vocab.get(ch)
        if tid is None:
            continue
        ids.append(tid)
        keep.append(s)
    return ids, keep


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    samples = [
        "انا شوفت علم مصر بيرفرف",
        "علم الفيزيا من اهم العلوم البشرية",
        "الشمس طالعة والجو حلو",
        "خالد شرب الشاي في القهوة",
    ]
    print(f"{'arabic':<45} {'romanized'}")
    print("-" * 80)
    for s in samples:
        latin, src = romanize_with_map(s)
        print(f"{s:<45} {latin}")
        assert len(latin) == len(src), "index map must match the output length"
        for i, j in enumerate(src):
            assert 0 <= j < len(s), f"source index {j} out of range for {s!r}"
    print()
    print("index map is consistent for every sample")
