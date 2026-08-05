"""Tests for the surprisal analysis.

The language model is never loaded here. ``collect_windows`` takes its scorer as
an argument, so a stub that returns one known value per word covers the part
that is easy to get wrong -- which word a subword belongs to, which position the
control is read at, and whether the two stay paired.
"""

import numpy as np
import pytest

from mend.surprisal import (
    WINDOW,
    _fold_to_words,
    _word_starts,
    bootstrap_ci,
    clean_position,
    collect_windows,
    matched_clean_position,
    paired_difference_ci,
    summarise_effects,
)


class StubScorer:
    """Returns a caller-supplied surprisal per word, keyed by sentence."""

    def __init__(self, table):
        self.table = table
        self.seen = []

    def word_surprisals(self, sentences, batch=16):
        self.seen.extend(sentences)
        return [np.asarray(self.table[s], dtype=float) for s in sentences]


# --------------------------------------------------------------------------
# folding subwords into words
# --------------------------------------------------------------------------


def test_word_starts_gives_character_spans():
    assert _word_starts("send it now") == [(0, 4), (5, 7), (8, 11)]


def test_subwords_sum_into_their_word():
    text = "ab cd"
    # "ab" arrives as two subwords, "cd" as one
    offsets = [(0, 1), (1, 2), (2, 5)]
    bits = np.array([1.0, 2.0, 4.0])
    assert _fold_to_words(text, offsets, bits).tolist() == [3.0, 4.0]


def test_a_leading_space_does_not_push_a_subword_into_the_previous_word():
    """The tokeniser folds the preceding space into the token, so a raw offset
    points at the end of the previous word."""
    text = "ab cd"
    offsets = [(0, 2), (2, 5)]  # the second token's span starts on the space
    bits = np.array([1.0, 5.0])
    assert _fold_to_words(text, offsets, bits).tolist() == [1.0, 5.0]


def test_folding_returns_one_value_per_word():
    text = "one two three four"
    offsets = [(0, 3), (3, 7), (7, 13), (13, 18)]
    out = _fold_to_words(text, offsets, np.ones(4))
    assert out.shape == (4,)


# --------------------------------------------------------------------------
# mapping the boundary onto the fluent control
# --------------------------------------------------------------------------


def test_clean_position_counts_surviving_tokens():
    # tokens 1 and 2 were injected, so a boundary at 3 sits at clean index 1
    assert clean_position([0, None, None, 1, 2], 3) == 1


def test_clean_position_is_zero_at_the_start():
    assert clean_position([None, None, 0], 0) == 0


def test_matched_position_agrees_when_the_order_is_preserved():
    assert matched_clean_position([0, None, None, 1, 2], 3) == 1


def test_matched_position_is_undefined_when_a_repair_arrives_late():
    """The injector's delayed repair emits a clean token after later ones, so
    counting survivors lands on an unrelated word and there is no match."""
    # clean index 2 has not been emitted yet at the boundary, but 3 and 4 have
    alignment = [0, 1, 3, 4, None, 2, 5]
    assert clean_position(alignment, 4) == 4        # the old answer
    assert matched_clean_position(alignment, 4) is None


def test_matched_position_holds_for_every_real_transcript_row():
    """Real speech is always in order, so the guard must never discard it."""
    alignment = [0, None, 1, 2, None, 3]
    for b in range(len(alignment)):
        assert matched_clean_position(alignment, b) == clean_position(alignment, b)


# --------------------------------------------------------------------------
# collecting windows
# --------------------------------------------------------------------------


def build_row(n_before=6, n_after=6):
    """A row whose single filler has a full window on both sides."""
    before = [f"w{i}" for i in range(n_before)]
    after = [f"x{i}" for i in range(n_after)]
    disfluent = before + ["um"] + after
    clean = before + after
    alignment = list(range(n_before)) + [None] + list(
        range(n_before, n_before + n_after)
    )
    return {
        "disfluent": " ".join(disfluent),
        "clean": " ".join(clean),
        "disfluent_tokens": disfluent,
        "clean_tokens": clean,
        "alignment": alignment,
        "annotations": [
            {"type": "filler", "interregnum": {"start": n_before, "end": n_before + 1}}
        ],
    }


def scorer_for(rows, disfluent_value=9.0, clean_value=1.0):
    table = {}
    for row in rows:
        table[row["disfluent"]] = [disfluent_value] * len(row["disfluent_tokens"])
        table[row["clean"]] = [clean_value] * len(row["clean_tokens"])
    return StubScorer(table)


def test_window_is_centred_on_the_boundary():
    row = build_row()
    data, control = collect_windows([row], scorer_for([row]), batch=4, verbose=False)
    assert data["filler"].shape == (1, 2 * WINDOW + 1)
    assert control["filler"].shape == (1, 2 * WINDOW + 1)


def test_control_is_read_from_the_fluent_sentence():
    row = build_row()
    data, control = collect_windows(
        [row], scorer_for([row], disfluent_value=9.0, clean_value=1.0),
        batch=4, verbose=False,
    )
    assert data["filler"][0].tolist() == [9.0] * (2 * WINDOW + 1)
    assert control["filler"][0].tolist() == [1.0] * (2 * WINDOW + 1)


def test_disfluency_without_a_full_window_is_dropped():
    """Only instances with ±5 real words on both sides are analysable."""
    row = build_row(n_before=2, n_after=6)  # boundary too close to the start
    data, _ = collect_windows([row], scorer_for([row]), batch=4, verbose=False)
    assert data == {}


def test_rows_with_no_annotations_are_skipped():
    row = build_row()
    row["annotations"] = []
    data, _ = collect_windows([row], scorer_for([row]), batch=4, verbose=False)
    assert data == {}


def test_unknown_annotation_types_are_skipped():
    row = build_row()
    row["annotations"][0]["type"] = "false_start"
    data, _ = collect_windows([row], scorer_for([row]), batch=4, verbose=False)
    assert data == {}


def test_each_distinct_sentence_is_scored_once():
    """Two disfluencies in one sentence must not cost two forward passes."""
    row = build_row()
    scorer = scorer_for([row])
    collect_windows([row, row], scorer, batch=4, verbose=False)
    assert len(scorer.seen) == len(set(scorer.seen))


def test_data_and_control_rows_stay_paired():
    """Row i of the control must belong to row i of the data, or the paired
    difference silently compares unrelated instances."""
    rows = [build_row(), build_row(n_before=7, n_after=7)]
    table = {}
    for i, row in enumerate(rows):
        table[row["disfluent"]] = [float(i)] * len(row["disfluent_tokens"])
        table[row["clean"]] = [float(i) + 100] * len(row["clean_tokens"])
    data, control = collect_windows(rows, StubScorer(table), batch=4, verbose=False)
    for i in range(len(rows)):
        assert data["filler"][i][0] == pytest.approx(control["filler"][i][0] - 100)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def test_bootstrap_mean_is_the_sample_mean():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    means, lo, hi = bootstrap_ci(data, n_draws=200)
    assert means.tolist() == [2.0, 3.0]
    assert (lo <= means).all() and (means <= hi).all()


def test_bootstrap_is_deterministic_for_a_seed():
    data = np.random.default_rng(0).normal(size=(50, 3))
    a = bootstrap_ci(data, n_draws=200, seed=7)[1]
    b = bootstrap_ci(data, n_draws=200, seed=7)[1]
    assert a.tolist() == b.tolist()


def test_paired_difference_is_the_ci_of_the_difference():
    data = np.random.default_rng(1).normal(size=(30, 3))
    ctrl = np.random.default_rng(2).normal(size=(30, 3))
    means, _, _ = paired_difference_ci(data, ctrl, n_draws=100)
    assert means == pytest.approx((data - ctrl).mean(axis=0))


def test_pairing_removes_variance_that_is_shared():
    """Each instance is its own baseline, which is the point of pairing."""
    shared = np.random.default_rng(3).normal(size=(200, 1)) * 10
    data = shared + 2.0
    ctrl = shared
    _, lo, hi = paired_difference_ci(data, ctrl, n_draws=300)
    assert (hi - lo)[0] < 0.5  # the shared noise cancels
    assert lo[0] > 0  # and the +2 effect survives


# --------------------------------------------------------------------------
# the reported effect
# --------------------------------------------------------------------------


def spike_at(position, height=5.0, n=40):
    """n instances whose profile is flat except for a spike at ``position``."""
    data = np.ones((n, 2 * WINDOW + 1))
    data[:, WINDOW + position] += height
    return data, np.ones((n, 2 * WINDOW + 1))


def test_peak_position_is_reported_relative_to_the_boundary():
    data, ctrl = spike_at(0)
    report = summarise_effects({"repair": data}, {"repair": ctrl}, n_draws=100)
    assert report["repair"]["peak_position"] == 0
    assert report["repair"]["peak_effect_bits"] == pytest.approx(5.0)


def test_a_peak_before_the_boundary_is_reported_as_negative():
    """The synthetic repair peak sits at -1, on the reparandum, and that
    offset is a finding about the generator rather than an error."""
    data, ctrl = spike_at(-1)
    report = summarise_effects({"repair": data}, {"repair": ctrl}, n_draws=100)
    assert report["repair"]["peak_position"] == -1


def test_boundary_effect_is_reported_even_when_the_peak_is_elsewhere():
    data, ctrl = spike_at(2)
    report = summarise_effects({"repair": data}, {"repair": ctrl}, n_draws=100)
    assert report["repair"]["peak_position"] == 2
    assert report["repair"]["effect_at_boundary_bits"] == pytest.approx(0.0)


def test_a_downward_effect_is_not_hidden_by_the_absolute_peak():
    data, ctrl = spike_at(0, height=-6.0)
    report = summarise_effects({"repair": data}, {"repair": ctrl}, n_draws=100)
    assert report["repair"]["peak_effect_bits"] == pytest.approx(-6.0)


def test_only_types_present_are_reported():
    data, ctrl = spike_at(0)
    report = summarise_effects({"repair": data}, {"repair": ctrl}, n_draws=100)
    assert set(report) == {"repair"}
    assert report["repair"]["n"] == 40
