"""Phoneme variants, spelling identity, and the partial-diacritic override.

Three things are pinned here, all measured against real CATT-ECA output rather
than assumed (see PHONOLOGY.md for the measurements):

1. The triple dot is phonemic and survives; the caret is noise and does not.
2. مدينة and مدينه are one word, but sukun on a final ة is a real /t/ contrast.
3. A user can correct a reading by typing some diacritics, without knowing any
   code numbers, and an unresolvable correction is reported rather than guessed.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from adaptts.text.diacritics import (  # noqa: E402
    N_VARIANTS,
    VARIANT_LETTER,
    VARIANT_NONE,
    VARIANT_TAA_T,
    ReadingLexicon,
    WordReadings,
    letter_variants,
    normalize_variant_marks,
    strip_diacritics,
    strip_junk,
    vowel_pattern,
    word_identity,
)

TRIPLE_DOT = "؞"


# --------------------------------------------------------------------------
# The triple dot is phonemic; the caret is not
# --------------------------------------------------------------------------


def test_triple_dot_survives_normalization_as_a_variant_mark():
    """Measured: ؞ attaches only to ق, ج, ف and marks a non-default reading.

    Verbatim CATT output for "technology" and "now". Discarding this loses the
    /zh/ in تكنولوجيا and the glottal stop in دلوقتي.
    """
    got = normalize_variant_marks("إِلتُّكْنُوْلُوْج" + TRIPLE_DOT + "ْيَا")
    assert "~" in got, f"variant mark lost: {got!r}"
    assert TRIPLE_DOT not in got, "the raw triple dot should be rewritten to ~"


def test_caret_is_deleted_because_it_marks_nothing():
    """Measured on و ع م ر ز ن ه د with no phonological pattern."""
    assert normalize_variant_marks("م^َصْرِ") == "مَصْرِ"
    assert normalize_variant_marks("الن^َّظَرِي^َّه") == "النَّظَرِيَّه"


def test_variant_mark_on_an_impossible_letter_is_dropped():
    """The mark applies to ق, ج and ف only, so anywhere else it is a stray."""
    assert "~" not in normalize_variant_marks("كْبِير" + TRIPLE_DOT + "َه")


def test_letter_variants_line_up_with_the_skeleton():
    """The model reads the skeleton, so the feature vector must match it 1:1."""
    for word in ["وَق~ْتِي", "ج~َامْعَه", "مَدِينَةْ", "عَلَمْ", "إِلْف~ِيْدْيُو"]:
        v = letter_variants(word)
        skeleton = strip_diacritics(word)
        assert len(v) == len(skeleton), (
            f"{word}: {len(v)} variants for {len(skeleton)} letters ({skeleton})"
        )
        assert all(0 <= x < N_VARIANTS for x in v), v


def test_variant_marks_never_create_a_reading():
    """CATT marks the qaf of قرار inconsistently between contexts.

    Promoting the mark to a pattern distinction would split that word on a
    diacritizer error, which is the failure mode that made الحكايه look
    ambiguous. The vowel pattern must therefore ignore variants entirely.
    """
    assert vowel_pattern("ق~َرَارْ") == vowel_pattern("قَرَارْ")
    assert strip_junk("ق~َرَارْ") == "قَرَارْ"


def test_variant_position_is_recorded_on_the_right_letter():
    v = letter_variants("وَق~ْتِي")          # و ق ت ي
    assert v[1] == VARIANT_LETTER, v
    assert v[0] == v[2] == v[3] == VARIANT_NONE, v


# --------------------------------------------------------------------------
# Taa marbuta: one identity, one real contrast
# --------------------------------------------------------------------------


def test_the_two_spellings_are_one_word():
    """مدينة and مدينه are the same word, written both ways in the corpus.

    Kept apart they become two entries with two code sets, splitting the
    training data; and CATT diacritizes the ة spelling well but the ه spelling
    badly, so the bad half would define its own readings.
    """
    for a, b in [("مدينة", "مدينه"), ("الحكاية", "الحكايه"), ("كبيرة", "كبيره")]:
        assert word_identity(a) == word_identity(b), f"{a} vs {b}"


def test_sukun_on_final_taa_marbuta_marks_the_t():
    """مَدِينَة is /madiina/; مَدِينَةْ is /madiinat/. A real contrast."""
    assert letter_variants("مَدِينَة")[-1] == VARIANT_NONE
    assert letter_variants("مَدِينَةْ")[-1] == VARIANT_TAA_T


def test_the_t_contrast_does_not_split_the_word():
    """It is a phoneme feature, not a reading: the pattern must be unaffected.

    Otherwise every feminine noun becomes a homograph on a diacritizer whim.
    """
    assert vowel_pattern("مَدِينَة") == vowel_pattern("مَدِينَةْ")


# --------------------------------------------------------------------------
# The override channel
# --------------------------------------------------------------------------


def _lexicon() -> ReadingLexicon:
    """مصر and علم ambiguous, مدينه not, as the real corpus gives them."""
    egypt, insist = "مَصْرْ", "مُصِرّ"
    sci, flag = "عِلْمْ", "عَلَمْ"
    return ReadingLexicon(
        {
            "مصر": WordReadings(
                word="مصر", patterns=[vowel_pattern(egypt), vowel_pattern(insist)],
                counts=[40, 18], examples=[egypt, insist], total=58,
            ),
            "علم": WordReadings(
                word="علم", patterns=[vowel_pattern(sci), vowel_pattern(flag)],
                counts=[18, 3], examples=[sci, flag], total=21,
            ),
            "مدينه": WordReadings(
                word="مدينه", patterns=[vowel_pattern("مَدِينَه")],
                counts=[30], examples=["مَدِينَه"], total=30,
            ),
        },
        max_codes=4,
    )


def test_the_users_own_example():
    """"انا كنت مُصِرّ على ان مصر ..." must pin the first word and free the second."""
    lex = _lexicon()
    code, _ = lex.code_from_partial_marks("مُصِرّ")
    assert code == 1, code
    code, _ = lex.code_from_partial_marks("مَصْرْ")
    assert code == 0, code


def test_undiacritized_input_is_left_to_the_model():
    """The system must not require diacritics; bare text means "you decide"."""
    lex = _lexicon()
    code, reason = lex.code_from_partial_marks("مصر")
    assert code == -1, (code, reason)


def test_one_interior_mark_is_enough():
    """Partial means partial: مصر's readings differ only at the interior letter."""
    lex = _lexicon()
    assert lex.code_from_partial_marks("مصِر")[0] == 1
    assert lex.code_from_partial_marks("مصْر")[0] == 0


def test_a_mark_the_lexicon_cannot_use_is_reported_not_guessed():
    """The word-initial position is discarded, because diacritizers pollute it.

    A user marking only there has said nothing the lexicon distinguishes. Saying
    so is better than picking a reading at random.
    """
    lex = _lexicon()
    code, reason = lex.code_from_partial_marks("مُصر")
    assert code == -1
    assert "initial" in reason or "distinguish" in reason, reason


def test_marks_that_match_no_reading_are_reported():
    lex = _lexicon()
    code, reason = lex.code_from_partial_marks("مصُر")   # neither sukun nor kasra
    assert code == -1, (code, reason)


def test_single_reading_words_need_no_marks():
    lex = _lexicon()
    assert lex.code_from_partial_marks("مدينه")[0] == 0


def test_lookup_folds_the_spelling():
    """A caller holding مدينة must find the entry stored under مدينه."""
    lex = _lexicon()
    assert lex.get("مدينة") is not None
    assert lex.n_codes("مدينة") == 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
