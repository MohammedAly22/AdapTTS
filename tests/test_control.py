"""Controllability: partial diacritics, the complexity view, and the knobs.

The design point these pin down: a user corrects a pronunciation by writing the
vowels, never by choosing a code number. Codes are assigned by corpus frequency,
so nobody can know that code 1 means مُصِرّ, and the answer changes when the
lexicon is rebuilt.

The system must also still work with no diacritics at all, which is the whole
premise, so the bare-input path is tested alongside the override path.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch  # noqa: E402

from adaptts.infer.pipeline import AdapTTS, split_user_diacritics  # noqa: E402
from adaptts.models.context_encoder import ContextEncoder  # noqa: E402
from adaptts.text.diacritics import (  # noqa: E402
    ReadingLexicon, WordReadings, vowel_pattern,
)
from adaptts.text.vocab import CharVocab  # noqa: E402
from adaptts.utils.config import load_config  # noqa: E402

FREE = "انا كنت مصر على ان مصر عندها امكانيات"
PINNED = "انا كنت مُصِرّ على ان مصر عندها امكانيات"

EGYPT, INSIST = "مَصْرْ", "مُصِرّ"


def _lexicon() -> ReadingLexicon:
    return ReadingLexicon(
        {"مصر": WordReadings(
            word="مصر", patterns=[vowel_pattern(EGYPT), vowel_pattern(INSIST)],
            counts=[40, 18], examples=[EGYPT, INSIST], total=58)},
        max_codes=4,
    )


def _tts() -> AdapTTS:
    cfg = load_config("configs/exp0_small.yaml")
    vocab = CharVocab.build([FREE, "علم الفيزيا", "مدينه كبيره"], min_freq=1)
    ctx = ContextEncoder(
        vocab_size=len(vocab), max_codes=cfg.discovery.max_codes_per_word,
        d_model=64, n_layers=2, n_heads=4, teacher_dim=cfg.teacher.hidden_size,
    ).eval()
    return AdapTTS(cfg, vocab, _lexicon(), ctx, None, None, torch.device("cpu"))


# --------------------------------------------------------------------------
# Diacritics never reach the model
# --------------------------------------------------------------------------


def test_the_model_never_sees_a_diacritic():
    """The premise of the system: input is undiacritized, always.

    A user's marks are read as instructions and removed, so the model's input
    distribution is identical whether or not they typed any.
    """
    bare, _, _, _ = split_user_diacritics(PINNED, _lexicon())
    assert bare == FREE, bare
    for mark in ("َ", "ُ", "ِ", "ّ", "ْ"):
        assert mark not in bare


def test_bare_input_leaves_every_decision_to_the_model():
    bare, overrides, unresolved, _ = split_user_diacritics(FREE, _lexicon())
    assert bare == FREE
    assert not overrides
    assert not unresolved


# --------------------------------------------------------------------------
# The override reaches the acoustic input
# --------------------------------------------------------------------------


def test_partial_diacritics_pin_one_occurrence_only():
    """The user's example: the first مصر is pinned, the second stays free."""
    tts = _tts()
    plan = tts.analyze(PINNED)
    pinned = next(w for w in plan.words if w.index == 2)
    assert pinned.code == 1, pinned.code
    assert pinned.from_user is True
    other = [w for w in plan.hard_words if w.index != 2]
    assert other and not any(w.from_user for w in other)


def test_the_override_reaches_the_tensor_the_acoustic_model_reads():
    """A plan that records an override but does not pass it on is decorative."""
    tts = _tts()
    plan = tts.analyze(PINNED)
    pc = plan.pc_per_char(tts.cfg.discovery.max_codes_per_word)
    widx = plan.inputs["word_index"]
    assert set(pc[widx == 2].tolist()) == {1}


def test_set_reading_takes_vowels_not_code_numbers():
    tts = _tts()
    plan = tts.analyze(FREE)
    plan.set_reading("مصر", "مُصِرّ")
    assert all(w.code == 1 for w in plan.words if w.word == "مصر")


def test_an_unresolvable_correction_raises_with_a_reason():
    tts = _tts()
    plan = tts.analyze(FREE)
    try:
        plan.set_reading("مصر", "مصُر")     # damma matches neither reading
    except ValueError as exc:
        assert "resolve" in str(exc), exc
    else:
        raise AssertionError("an impossible reading was accepted")


# --------------------------------------------------------------------------
# The complexity view
# --------------------------------------------------------------------------


def test_complexity_table_covers_every_word_and_names_its_source():
    tts = _tts()
    plan = tts.analyze(PINNED)
    rows = plan.complexity_table()
    assert len(rows) == len(plan.words)
    for r in rows:
        assert set(r) >= {"word", "readings", "difficulty", "depth", "source"}
    sources = {r["source"] for r in rows}
    assert "user" in sources          # the pinned word
    assert "unambiguous" in sources   # the single-reading words


def test_single_reading_words_are_cheap():
    """The adaptive claim, made falsifiable.

    A word with one known reading has nothing to disambiguate, so its difficulty
    must be lower than an ambiguous word's. If this ever fails, the difficulty
    head is not tracking ambiguity and the adaptive-depth story is empty.
    """
    tts = _tts()
    plan = tts.analyze(PINNED)
    rows = plan.complexity_table()
    easy = [r["difficulty"] for r in rows if r["readings"] == 1]
    hard = [r["difficulty"] for r in rows if r["readings"] > 1]
    assert easy and hard
    assert max(easy) <= min(hard), (easy, hard)


def test_complexity_report_is_printable():
    tts = _tts()
    text = tts.analyze(PINNED).complexity_report()
    assert "difficulty" in text and "مصر" in text


# --------------------------------------------------------------------------
# The knobs
# --------------------------------------------------------------------------


def test_controls_reject_values_outside_their_range():
    """A silently clamped tempo is worse than a refused one."""
    tts = _tts()
    plan = tts.analyze(FREE)
    for setter, bad in [
        (plan.set_tempo, 99.0),
        (plan.set_cfg_scale, -1.0),
        (plan.set_temperature, 5.0),
        (plan.set_budget, 0.0),
    ]:
        try:
            setter(bad)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError(f"{setter.__name__} accepted {bad}")


def test_controls_accept_sensible_values():
    tts = _tts()
    plan = tts.analyze(FREE)
    plan.set_tempo(1.25).set_cfg_scale(2.0).set_temperature(0.8)
    assert plan.tempo == 1.25 and plan.cfg_scale == 2.0 and plan.temperature == 0.8


def test_budget_caps_depth_without_forcing_it():
    """A budget is a ceiling, so it can only lower the depth, never raise it."""
    tts = _tts()
    plan = tts.analyze(FREE)
    before = plan.depth
    plan.set_budget(0.5)
    assert plan.depth <= before
    assert plan.depth in tts.cfg.acoustic.exit_layers


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
