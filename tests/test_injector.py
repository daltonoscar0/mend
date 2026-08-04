"""Tests for the disfluency injector."""

import random

import pytest

from mend.injector import (
    FILLERS,
    INTERREGNA,
    SWAP_GROUPS,
    InjectionConfig,
    InjectionResult,
    inject,
    remove_disfluencies,
    spoken_form,
)

SENTENCES = [
    "Send the quarterly report to Sarah before Friday.",
    "Remind me to call Michael about the invoice on Tuesday morning.",
    "The committee published its findings in October, and the response was immediate.",
    "Schedule the review for 14 March and book the large meeting room.",
    "Move the blue folder from the kitchen to the office, please.",
    "Tell David that the contract needs three more signatures.",
    "I left the draft on your desk; have a look when you get in.",
    "Book a table for six people at seven o'clock on Thursday.",
]

# Rates high enough that most sentences receive several disfluencies.
AGGRESSIVE = InjectionConfig(
    filler_rate=0.9,
    repetition_rate=0.9,
    repair_rate=0.95,
    false_start_rate=0.5,
    interregnum_rate=0.6,
)


def all_results(config=AGGRESSIVE, n_seeds=60):
    for seed in range(n_seeds):
        for sent in SENTENCES:
            yield sent, inject(sent, config, rng_seed=seed)


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_same_seed_gives_identical_output():
    for sent in SENTENCES:
        a = inject(sent, AGGRESSIVE, rng_seed=7)
        b = inject(sent, AGGRESSIVE, rng_seed=7)
        assert a.to_dict() == b.to_dict()


def test_injection_does_not_disturb_global_random_state():
    random.seed(123)
    expected = [random.random() for _ in range(3)]
    random.seed(123)
    inject(SENTENCES[0], AGGRESSIVE, rng_seed=99)
    assert [random.random() for _ in range(3)] == expected


def test_different_seeds_usually_differ():
    outputs = {inject(SENTENCES[0], AGGRESSIVE, rng_seed=s).disfluent_text for s in range(30)}
    assert len(outputs) > 5


def test_round_trip_survives_serialisation():
    result = inject(SENTENCES[1], AGGRESSIVE, rng_seed=3)
    restored = InjectionResult.from_dict(result.to_dict())
    assert restored.to_dict() == result.to_dict()
    assert remove_disfluencies(restored) == result.clean_text


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------


def test_removing_annotated_spans_recovers_clean_text():
    for sent, result in all_results():
        assert remove_disfluencies(result) == result.clean_text, sent


def test_clean_text_matches_source_sentence():
    for sent, result in all_results(n_seeds=3):
        assert result.clean_text == " ".join(sent.split())


def test_unannotated_tokens_are_exactly_the_clean_tokens():
    """Every disfluent token is either injected material or a clean token."""
    for _, result in all_results():
        dropped = set()
        for ann in result.annotations:
            dropped.update(ann.deleted_indices())
        kept = [
            (result.alignment[i], result.disfluent_tokens[i])
            for i in range(len(result.disfluent_tokens))
            if i not in dropped
        ]
        kept.sort(key=lambda pair: pair[0])
        assert [tok for _, tok in kept] == [spoken_form(t) for t in result.clean_tokens]


# --------------------------------------------------------------------------
# span integrity
# --------------------------------------------------------------------------


def test_spans_are_within_bounds_and_non_empty():
    for _, result in all_results():
        n = len(result.disfluent_tokens)
        for ann in result.annotations:
            for span in (ann.reparandum, ann.interregnum, ann.repair):
                if span is None:
                    continue
                assert 0 <= span.start < span.end <= n, ann


def test_deleted_spans_never_overlap_each_other():
    for _, result in all_results():
        seen = set()
        for ann in result.annotations:
            idx = set(ann.deleted_indices())
            assert not (idx & seen), result.disfluent_text
            seen |= idx


def test_deleted_tokens_are_exactly_the_injected_ones():
    for _, result in all_results():
        dropped = set()
        for ann in result.annotations:
            dropped.update(ann.deleted_indices())
        injected = {i for i, src in enumerate(result.alignment) if src is None}
        assert dropped == injected


def test_alignment_covers_every_clean_token_exactly_once():
    """A delayed repair may put its correction out of order, but never twice."""
    for _, result in all_results():
        sources = [s for s in result.alignment if s is not None]
        assert sorted(sources) == list(range(len(result.clean_tokens)))


def test_only_delayed_repairs_break_the_spoken_order():
    for _, result in all_results():
        sources = [s for s in result.alignment if s is not None]
        if sources == sorted(sources):
            continue
        # the sole reason for reordering is a repair whose correction is
        # separated from its reparandum by intervening words
        gapped = [
            ann
            for ann in result.annotations
            if ann.type == "repair"
            and ann.interregnum is not None
            and ann.reparandum.end < ann.interregnum.start
        ]
        assert gapped, result.disfluent_text


# --------------------------------------------------------------------------
# per-type structure
# --------------------------------------------------------------------------


def _tokens(result, span):
    return result.disfluent_tokens[span.start : span.end]


def test_filler_annotations_carry_real_fillers():
    known = {" ".join(f) for f in FILLERS}
    seen = 0
    for _, result in all_results():
        for ann in result.annotations:
            if ann.type != "filler":
                continue
            seen += 1
            assert ann.reparandum is None and ann.repair is None
            assert " ".join(_tokens(result, ann.interregnum)) in known
    assert seen > 0


def test_repetition_reparandum_equals_repair():
    seen = 0
    for _, result in all_results():
        for ann in result.annotations:
            if ann.type != "repetition":
                continue
            seen += 1
            assert ann.interregnum is None
            assert _tokens(result, ann.reparandum) == _tokens(result, ann.repair)
    assert seen > 0


def test_repair_reparandum_differs_from_its_repair():
    known_interregna = {" ".join(x) for x in INTERREGNA}
    seen = 0
    for _, result in all_results():
        for ann in result.annotations:
            if ann.type != "repair":
                continue
            seen += 1
            assert _tokens(result, ann.reparandum) != _tokens(result, ann.repair)
            if ann.interregnum is not None:
                assert " ".join(_tokens(result, ann.interregnum)) in known_interregna
    assert seen > 0


def test_single_token_repairs_swap_within_a_category():
    groups = [set(g) for g in SWAP_GROUPS]
    seen = 0
    for _, result in all_results():
        for ann in result.annotations:
            if ann.type != "repair" or len(ann.reparandum) != 1 or len(ann.repair) != 1:
                continue
            seen += 1
            wrong = _tokens(result, ann.reparandum)[0]
            right = _tokens(result, ann.repair)[0]
            same_group = any(wrong in g and right in g for g in groups)
            assert same_group or (wrong.isdigit() and right.isdigit())
    assert seen > 0


def test_repair_span_holds_the_correct_words():
    for _, result in all_results():
        for ann in result.annotations:
            if ann.type != "repair":
                continue
            for offset in range(len(ann.repair)):
                clean_idx = result.alignment[ann.repair.start + offset]
                assert clean_idx is not None
                assert _tokens(result, ann.repair)[offset] == spoken_form(
                    result.clean_tokens[clean_idx]
                )


def test_multi_token_repairs_overlap_their_reparandum():
    """The point of a span repair: both halves share most of their words."""
    seen = 0
    for _, result in all_results():
        for ann in result.annotations:
            if ann.type != "repair" or len(ann.repair) < 2:
                continue
            seen += 1
            wrong = set(_tokens(result, ann.reparandum))
            right = set(_tokens(result, ann.repair))
            assert wrong & right, result.disfluent_text
    assert seen > 0


def test_delayed_repairs_put_words_between_the_two_halves():
    seen = 0
    for _, result in all_results():
        for ann in result.annotations:
            if ann.type != "repair" or ann.interregnum is None:
                continue
            if ann.reparandum.end == ann.interregnum.start:
                continue
            seen += 1
            # the intervening tokens are ordinary clean words, kept as they are
            between = range(ann.reparandum.end, ann.interregnum.start)
            assert all(result.alignment[i] is not None for i in between)
            # and the correction belongs where the reparandum stood
            assert result.alignment[ann.repair.start] < result.alignment[ann.reparandum.end]
    assert seen > 0


def test_false_start_precedes_the_restart():
    seen = 0
    at_start = 0
    mid_sentence = 0
    for _, result in all_results():
        for ann in result.annotations:
            if ann.type != "false_start":
                continue
            seen += 1
            assert ann.repair.start == ann.reparandum.end
            resumes_at = result.alignment[ann.repair.start]
            assert resumes_at is not None
            if ann.reparandum.start == 0:
                at_start += 1
                assert resumes_at == 0
            else:
                mid_sentence += 1
                # a mid-sentence restart follows a clause boundary
                previous = result.clean_tokens[resumes_at - 1]
                assert previous.rstrip("\"”'’)").endswith((",", ";", ":"))
    assert seen > 0
    assert at_start > 0 and mid_sentence > 0


def test_boundary_index_points_at_the_interruption():
    for _, result in all_results():
        for ann in result.annotations:
            b = ann.boundary_index()
            assert b is not None
            if ann.interregnum is not None:
                assert b == ann.interregnum.start
            elif ann.repair is not None:
                assert b == ann.repair.start


# --------------------------------------------------------------------------
# punctuation degradation
# --------------------------------------------------------------------------


def test_disfluent_side_is_lowercase_and_unpunctuated():
    for _, result in all_results():
        text = result.disfluent_text
        assert text == text.lower()
        assert not any(c in text for c in '.,;:!?"')


def test_apostrophes_inside_words_survive():
    result = inject("Don't forget Sarah's report.", AGGRESSIVE, rng_seed=1)
    assert "don't" in result.disfluent_tokens
    assert "sarah's" in result.disfluent_tokens


def test_degradation_can_be_switched_off():
    cfg = InjectionConfig(
        filler_rate=0.0,
        repetition_rate=0.0,
        repair_rate=0.0,
        false_start_rate=0.0,
        lowercase=False,
        strip_punctuation=False,
    )
    result = inject(SENTENCES[0], cfg, rng_seed=0)
    assert result.disfluent_text == result.clean_text


# --------------------------------------------------------------------------
# edge cases
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "—", "... ---"])
def test_degenerate_input_is_handled(text):
    result = inject(text, AGGRESSIVE, rng_seed=0)
    assert result.disfluent_text == ""
    assert result.clean_text == ""


def test_punctuation_only_tokens_are_dropped_from_the_target():
    result = inject("Call Sarah — the report is late.", AGGRESSIVE, rng_seed=4)
    assert "—" not in result.clean_text
    assert remove_disfluencies(result) == result.clean_text


def test_zero_rates_leave_the_words_alone():
    cfg = InjectionConfig(
        filler_rate=0.0, repetition_rate=0.0, repair_rate=0.0, false_start_rate=0.0
    )
    for sent in SENTENCES:
        result = inject(sent, cfg, rng_seed=11)
        assert result.annotations == []
        assert result.disfluent_tokens == [spoken_form(t) for t in result.clean_tokens]
