"""Tests for Egyptian Arabic text normalization.

The headline rule under test: Egyptian Arabic does not insert a linking waw
between magnitude groups. 2024 is "الفين اربعة و عشرين", never
"الفين و اربعة و عشرين". The waw belongs to the tens group alone.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from adaptts.text.egyptian import (  # noqa: E402
    date_to_words,
    decimal_to_words,
    digits_one_by_one,
    expand_abbreviations,
    normalize_egyptian,
    number_to_words,
    ordinal_to_words,
    time_to_words,
)


# --------------------------------------------------------------------------
# The number rule
# --------------------------------------------------------------------------


def test_no_waw_between_magnitude_groups():
    """The rule from the brief, stated as an executable assertion."""
    assert number_to_words(2024) == "الفين اربعة و عشرين"
    assert number_to_words(876) == "تمنمية ستة و سبعين"
    assert number_to_words(1986) == "الف تسعمية ستة و تمانين"

    # The wrong (Modern Standard) forms must never appear.
    for n in (2024, 876, 1986, 1500, 3021):
        said = number_to_words(n)
        assert " و " not in said.split(" و ")[0] or True  # structural check below
    # Precisely: a waw may only be followed by a tens word.
    tens = {"عشرين", "تلاتين", "اربعين", "خمسين", "ستين", "سبعين", "تمانين", "تسعين"}
    for n in list(range(0, 300)) + [876, 1986, 2024, 12345, 999999]:
        parts = number_to_words(n).split()
        for i, tok in enumerate(parts):
            if tok == "و":
                assert i + 1 < len(parts), f"dangling waw in {n}: {parts}"
                assert parts[i + 1] in tens, (
                    f"{n} -> {' '.join(parts)}: waw must precede a tens word, "
                    f"got {parts[i + 1]!r}"
                )


def test_basic_cardinals():
    cases = {
        0: "صفر", 1: "واحد", 2: "اتنين", 10: "عشرة", 11: "حداشر",
        15: "خمستاشر", 20: "عشرين", 21: "واحد و عشرين", 99: "تسعة و تسعين",
        100: "مية", 200: "متين", 300: "تلتمية", 800: "تمنمية",
        1000: "الف", 2000: "الفين", 3000: "تلاتة الاف",
        1000000: "مليون", 2000000: "مليونين", 5000000: "خمسة ملايين",
    }
    for n, expected in cases.items():
        assert number_to_words(n) == expected, f"{n} -> {number_to_words(n)}"


def test_every_number_under_10000_is_speakable():
    """No crashes, no empty output, no leftover digits anywhere in the range."""
    for n in range(0, 10000):
        said = number_to_words(n)
        assert said, f"{n} produced nothing"
        assert not any(c.isdigit() for c in said), f"{n} -> {said} still has digits"


def test_negative_and_decimal():
    assert number_to_words(-5).startswith("سالب")
    assert decimal_to_words("3.5") == "تلاتة فاصلة خمسة"
    assert "فاصلة" in decimal_to_words("0.25")


def test_ordinals_and_digit_strings():
    assert ordinal_to_words(1) == "الاول"
    assert ordinal_to_words(3) == "التالت"
    assert ordinal_to_words(3, feminine=True) == "التالتة"
    assert digits_one_by_one("102") == "واحد صفر اتنين"
    assert digits_one_by_one("102", zero="زيرو") == "واحد زيرو اتنين"


# --------------------------------------------------------------------------
# Dates and times
# --------------------------------------------------------------------------


def test_dates_and_times():
    """Hours are plain cardinals, not ordinals.

    Egyptian speakers say "تمانية الا ربع", not the Modern Standard
    "الساعة الثامنة الا ربع".
    """
    assert date_to_words(15, 3, 2024) == "خمستاشر مارس الفين اربعة و عشرين"
    assert time_to_words(3, 30) == "تلاتة و نص"
    assert time_to_words(7, 45) == "تمانية الا ربع"
    assert time_to_words(1, 15) == "واحدة و ربع"
    assert time_to_words(12, 0) == "اتناشر"
    assert time_to_words(8, 20) == "تمانية و تلت"
    assert time_to_words(10, 40) == "حداشر الا تلت"


def test_time_says_alsaaa_only_once():
    """الساعة appears only where the writer typed it."""
    out = normalize_egyptian("الساعة 3:30")
    assert out.count("الساعة") == 1, out
    assert "تلاتة" in out and "نص" in out
    # And the ordinal form must not come back.
    assert "التالتة" not in out, out

    bare = normalize_egyptian("نتقابل 7:45")
    assert "الساعة" not in bare, bare
    assert "تمانية الا ربع" in bare, bare


# --------------------------------------------------------------------------
# Abbreviations
# --------------------------------------------------------------------------


def test_abbreviations_respect_word_boundaries():
    """الخصم must not be mangled by the الخ abbreviation."""
    out = normalize_egyptian("الخصم 25%")
    assert "الخصم" in out, out
    assert "الى اخره" not in out, out

    # But a genuine standalone abbreviation still expands.
    assert "الى اخره" in normalize_egyptian("كتب واقلام الخ")
    assert "دكتور" in normalize_egyptian("د. احمد")
    assert "مهندس" in normalize_egyptian("م. سمير")


# --------------------------------------------------------------------------
# Structured spans
# --------------------------------------------------------------------------


def test_thousands_separators_are_not_read_as_decimals():
    """2,500,000 is two and a half million, not "two point five"."""
    out = normalize_egyptian("ارباح 2,500,000 جنيه")
    assert "مليونين" in out, out
    assert "فاصلة" not in out, out


def test_fractions_use_the_plural_denominator():
    assert "ارباع" in normalize_egyptian("3/4 من الطلاب")
    assert normalize_egyptian("خد 1/2 كيلو").count("نص") == 1


def test_urls_and_emails():
    out = normalize_egyptian("شوف www.youtube.com")
    assert "يوتيوب" in out and "دوت" in out and "كوم" in out
    assert "www" not in out and "نقطة" not in out

    out = normalize_egyptian("ابعت على ahmed@gmail.com")
    assert "ات" in out and "جيميل" in out
    assert "@" not in out


def test_percent_and_currency():
    assert "في المية" in normalize_egyptian("خصم 25%")
    assert "دولار" in normalize_egyptian("السعر $50")
    assert "جنيه" in normalize_egyptian("السعر 250 ج.م")


def test_phone_numbers_are_grouped_not_spelled_out():
    """Egyptian mobile numbers are read in groups, with the prefix as a number.

    Nobody reads all eleven digits separately. 010 is "زيرو عشرة", and the
    eight subscriber digits break 3+2+3.
    """
    from adaptts.text.egyptian import phone_to_words

    assert phone_to_words("01027756313") == (
        "زيرو عشرة ، اتنين سبعة سبعة ، خمسة ستة ، تلاتة واحد تلاتة"
    )
    assert phone_to_words("01116953882").startswith("زيرو حداشر")
    # A fourth digit of 0 is absorbed into the prefix: 0100 -> زيرو مية.
    assert phone_to_words("01002776313").startswith("زيرو مية")
    assert "زيرو عشرة" not in phone_to_words("01002776313")

    # The prefix must never be spelled out digit by digit.
    out = normalize_egyptian("رقمي 01012345678")
    assert "زيرو عشرة" in out, out
    assert "صفر واحد صفر" not in out, out
    assert not any(c.isdigit() for c in out)


def test_phone_groups_can_be_read_as_numbers():
    from adaptts.text.egyptian import phone_to_words

    out = phone_to_words("01027756313", style="numbers")
    assert out.startswith("زيرو عشرة")
    assert "متين سبعة و سبعين" in out, out


def test_email_uses_dot_and_reads_names_as_names():
    """دوت, the borrowed word, not the Arabic نقطة. And a name is a name."""
    out = normalize_egyptian("ابعتلي على ahmed@gmail.com")
    assert "احمد" in out, out
    assert "ات" in out and "جيميل" in out and "دوت" in out, out
    assert "نقطة" not in out, out
    # The old behaviour spelled the name out letter by letter.
    assert "ايه اتش ام" not in out, out

    multi = normalize_egyptian("ابعت لـ mohamed.ali@company.com")
    assert "محمد" in multi and "علي" in multi, multi


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------


def test_output_never_contains_digits_or_latin():
    """Whatever goes in, the acoustic model must receive only speakable Arabic."""
    hard = [
        "الاجتماع يوم 15/3/2024 الساعة 10:30 ص في قاعة 5",
        "اتصل على 01098765432 او ابعت لـ info@company.com.eg",
        "الشركة حققت ارباح 2,500,000 جنيه بزيادة 15% عن 2023",
        "العنوان 25 شارع التحرير الدور ال3 شقة 12 القاهرة",
        "حمل التطبيق من www.google.com/play وقيمه 5 نجوم",
        "السعر $50 يعني حوالي 2400 ج.م تقريبا",
        "درجة الحرارة 30° والرطوبة 60%",
        "CEO الشركة قال ان الـ GDP زاد 3.2%",
    ]
    for text in hard:
        out = normalize_egyptian(text)
        bad = [c for c in out if c.isdigit() or (c.isascii() and c.isalpha())]
        assert not bad, f"{text!r} -> {out!r} leaked {sorted(set(bad))}"
        assert out.strip(), f"{text!r} normalized to nothing"


def test_diacritics_are_stripped_and_arabic_digits_converted():
    assert normalize_egyptian("عَلَم مِصر") == "علم مصر"
    assert "الف تسعمية ستة و تمانين" in normalize_egyptian("سنة ١٩٨٦")


def test_normalization_is_idempotent():
    """Running twice must not change the result, or cached text drifts."""
    for text in ["السنة دي 2024", "الساعة 3:30", "خصم 25%", "علم مصر بيرفرف"]:
        once = normalize_egyptian(text)
        assert normalize_egyptian(once) == once, f"{text!r}: {once!r}"


def test_empty_and_whitespace_input():
    assert normalize_egyptian("") == ""
    assert normalize_egyptian("   ") == ""
    assert normalize_egyptian("!!!") == ""


def test_elongation_is_collapsed():
    assert normalize_egyptian("جااااااامد") == "جاامد"


# --------------------------------------------------------------------------
# Romanization (used only to drive CTC forced alignment)
# --------------------------------------------------------------------------


def test_romanization_index_map_is_consistent():
    """Every Latin character must point at a real source character.

    The aligner converts Latin frame spans back into Arabic word spans through
    this map, so an off-by-one here silently misaligns every word.
    """
    from adaptts.text.romanize import romanize_with_map

    for text in [
        "انا شوفت علم مصر بيرفرف",
        "علم الفيزيا من اهم العلوم البشرية",
        "خالد شرب الشاي في القهوة",
        "الشمس طالعة والجو حلو",
    ]:
        latin, src = romanize_with_map(text)
        assert len(latin) == len(src), f"{text!r}: {len(latin)} vs {len(src)}"
        for i, j in enumerate(src):
            assert 0 <= j < len(text), f"source index {j} out of range"
        # Indices must be non-decreasing, or spans would cross word boundaries.
        assert all(a <= b for a, b in zip(src, src[1:])), "index map is not monotonic"
        # Spaces survive so word boundaries are recoverable.
        assert latin.count(" ") == text.count(" ")


def test_romanization_emits_only_aligner_vocabulary():
    """The MMS aligner has 31 tokens; anything else would be dropped silently."""
    from adaptts.text.romanize import romanize

    allowed = set("aieonutsrmkldghybpwcvjzf'qx ")
    for text in ["انا شوفت علم مصر", "خالد شرب الشاي", "الجو حلو النهارده"]:
        out = romanize(text)
        extra = set(out) - allowed
        assert not extra, f"{text!r} -> {out!r} emitted {sorted(extra)}"


def test_ctc_targets_drop_unmappable_characters():
    from adaptts.text.romanize import build_ctc_targets

    vocab = {c: i + 4 for i, c in enumerate("aieonutsrmkldghybpwcvjzf'qx")}
    ids, src = build_ctc_targets("انا شوفت علم", vocab, unk_id=3)
    assert len(ids) == len(src)
    assert ids, "produced no targets at all"
    assert all(i in vocab.values() for i in ids), "emitted an id outside the vocabulary"


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
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
