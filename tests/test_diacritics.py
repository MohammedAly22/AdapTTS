"""Tests for diacritizer-derived pronunciation labels.

This module replaced unsupervised acoustic clustering, which failed on the real
corpus by returning the channel's subscribe pitch as "homographs". The tests
below pin the two properties that matter: genuine readings separate, and
diacritizer artifacts do not create fake ones.

Every diacritized form here was produced by CATT-ECA on real corpus sentences.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from adaptts.text.diacritics import (  # noqa: E402
    ReadingLexicon,
    align_tokens,
    clean_marks,
    collect_readings,
    letter_marks,
    match_key,
    strip_diacritics,
    strip_junk,
    vowel_pattern,
)


# --------------------------------------------------------------------------
# Junk from the diacritizer
# --------------------------------------------------------------------------


def test_strip_junk_removes_the_characters_catt_emits():
    """Measured on real output: ^, the Arabic triple dot, and stray angle brackets."""
    dirty = "عَلَمْ^ مَصْر؞ْ >دول< انا…"
    clean = strip_junk(dirty)
    for ch in "^؞><…":
        assert ch not in clean, f"{ch!r} survived: {clean!r}"
    # The words themselves must be intact.
    assert "عَلَمْ" in clean and "دول" in clean and "انا" in clean


def test_strip_junk_keeps_letters_and_diacritics():
    text = "عِلْمْ الْفِيْزْيَا"
    assert strip_junk(text) == text


# --------------------------------------------------------------------------
# Vowel patterns
# --------------------------------------------------------------------------


def test_letter_marks_groups_diacritics_with_their_letter():
    """Marks bind to the preceding letter; a flat list confuses positions."""
    pairs = letter_marks("عَلَمْ")
    assert [l for l, _ in pairs] == ["ع", "ل", "م"]
    assert pairs[0][1] and pairs[1][1], "both fathas must be attached"


def test_real_homographs_get_different_patterns():
    """The whole system exists for this distinction."""
    cases = [
        ("عَلَمْ", "عِلْمْ", "علم: flag vs science"),
        ("مَصْرِ", "مُصِرِّ", "مصر: Egypt vs insisting"),
        ("دولْ", "دِوَلْ", "دول: these vs countries"),
        ("كَتَبْ", "كُتُبْ", "كتب: wrote vs books"),
        ("عَالَمْ", "عَالِمْ", "عالم: world vs scholar"),
    ]
    for a, b, label in cases:
        assert vowel_pattern(a) != vowel_pattern(b), (
            f"{label}: {a} and {b} collapsed to {vowel_pattern(a)!r}"
        )


def test_diacritizer_artifacts_collapse_to_one_pattern():
    """Artifacts are phonological impossibilities, not readings.

    Each pair below appeared in real CATT output for the same word. They differ
    only by a mark that cannot encode a pronunciation: a sukun on the initial
    letter, a sukun on a long vowel, or a mark inside the invariant article.
    Before clean_marks these produced 912 "ambiguous" word types on 3000
    sentences; after, 145.
    """
    cases = [
        ("فِيْ", "فْيْ", "في"),
        ("اللِّيْ", "اْللِّيْ", "اللي"),
        ("انِّ", "اْنّْ", "ان"),
        ("قَالْ", "قَاْلْ", "قال"),
        ("النَّاسْ", "اْلنَّاسْ", "الناس"),
        ("الْعَالَمْ", "اْلْعَالَمْ", "العالم"),
    ]
    for a, b, label in cases:
        assert vowel_pattern(a) == vowel_pattern(b), (
            f"{label}: {a} and {b} still differ "
            f"({vowel_pattern(a)!r} vs {vowel_pattern(b)!r})"
        )


def test_case_ending_is_ignored():
    """Egyptian speech drops case endings, so مَصْرِ and مَصْرْ are one reading.

    Keeping them apart would make a homograph of every noun in the corpus.
    """
    assert vowel_pattern("مَصْرِ") == vowel_pattern("مَصْرْ") == vowel_pattern("مَصْرُ")


def test_shadda_is_preserved_because_doubling_is_phonemic():
    assert vowel_pattern("مُصِرِّ") != vowel_pattern("مُصِرِ")


def test_clean_marks_never_empties_a_word():
    for w in ["عَلَمْ", "عِلْمْ", "اْللِّيْ", "ا", "فِيْ"]:
        assert clean_marks(letter_marks(w)), f"{w} cleaned to nothing"


# --------------------------------------------------------------------------
# Token alignment
# --------------------------------------------------------------------------


def test_match_key_folds_hamza_so_tokens_align():
    """CATT adds hamza the transcript lacks; that is orthography, not a reading."""
    for plain, diac in [
        ("انا", "أنَاْ"),
        ("الدول", "إِلدِّوَلْ"),
        ("السلام", "ألسَّلَامُ"),
        ("الناس", "إِلنَّاسْ"),
    ]:
        assert match_key(plain) == match_key(diac), f"{plain} vs {diac}"


def test_align_tokens_pairs_words_with_their_diacritized_form():
    plain = "انا شوفت علم مصر".split()
    diac = "أنَاْ شُوْفْتِ عَلَمْ مَصْرِ".split()
    out = align_tokens(plain, diac)
    assert out == diac, out


def test_align_tokens_returns_none_rather_than_mislabelling():
    """A mismatch must skip the occurrence, never guess."""
    plain = "انا شوفت علم".split()
    diac = "أنَاْ حَاجَةْ تَانْيَة".split()
    out = align_tokens(plain, diac)
    assert out[0] is not None          # انا matches
    assert out[1] is None and out[2] is None


def test_strip_diacritics_recovers_the_bare_word():
    assert strip_diacritics("عَلَمْ") == "علم"
    assert strip_diacritics("مُصِرِّ") == "مصر"


# --------------------------------------------------------------------------
# Collecting readings
# --------------------------------------------------------------------------


def _corpus():
    """A small corpus where علم is genuinely ambiguous and الجرس is not."""
    flag = [(["شوفت", "علم", "مصر"], ["شُوْفْتِ", "عَلَمْ", "مَصْرِ"])] * 5
    sci = [(["علم", "الفيزيا"], ["عِلْمْ", "الْفِيْزْيَا"])] * 5
    promo = [(["فعل", "الجرس"], ["فَعَّلْ", "الْجَرَسْ"])] * 8
    return flag + sci + promo


def test_collect_readings_finds_the_real_homograph():
    r = collect_readings(_corpus(), min_word_freq=3, min_pattern_count=2,
                         min_pattern_frac=0.2)
    assert "علم" in r, sorted(r)
    assert r["علم"].n_codes == 2, r["علم"].patterns


def test_collect_readings_leaves_unambiguous_words_alone():
    """The promo words that unsupervised clustering wrongly split."""
    r = collect_readings(_corpus(), min_word_freq=3, min_pattern_count=2,
                         min_pattern_frac=0.2)
    assert r["الجرس"].n_codes == 1
    assert r["مصر"].n_codes == 1


def test_a_single_odd_diacritization_is_not_a_reading():
    """One slip out of many uses is an error, not a homograph."""
    corpus = [(["كلمه"], ["كِلْمَة"])] * 20 + [(["كلمه"], ["كَلِمَة"])]
    r = collect_readings(corpus, min_word_freq=5, min_pattern_count=3,
                         min_pattern_frac=0.08)
    assert r["كلمه"].n_codes == 1, r["كلمه"].patterns


def test_reading_lexicon_roundtrips(tmp_path=None):
    import tempfile
    from pathlib import Path

    r = collect_readings(_corpus(), min_word_freq=3, min_pattern_count=2,
                         min_pattern_frac=0.2)
    lex = ReadingLexicon(r, max_codes=4)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "lex.json"
        lex.save(p)
        back = ReadingLexicon.load(p)
    assert back.ambiguous_words == lex.ambiguous_words
    assert back.n_codes("علم") == 2
    assert back.n_codes("الجرس") == 1
    assert back.n_codes("كلمة_غير_موجودة") == 1  # unseen word: one reading


def test_code_of_maps_an_occurrence_to_its_reading():
    r = collect_readings(_corpus(), min_word_freq=3, min_pattern_count=2,
                         min_pattern_frac=0.2)
    lex = ReadingLexicon(r, max_codes=4)
    a = lex.code_of("علم", "عَلَمْ")
    b = lex.code_of("علم", "عِلْمْ")
    assert a >= 0 and b >= 0 and a != b, (a, b)
    # An unseen pattern must report -1 rather than guessing.
    assert lex.code_of("علم", "عُلِمْ") == -1


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
