"""Tests for the scoring functions.

Every number in the results table comes out of these, and most of them are
count-based rather than position-based for reasons that are easy to undo by
accident. Nothing here loads a model.
"""

import pytest

from mend.evaluate import (
    copy_baseline,
    exact_match,
    filler_removal_counts,
    filler_vocab_for,
    repair_resolved,
    score,
    word_tokens,
)


def span(start, end):
    return {"start": start, "end": end}


def row(disfluent, clean, annotations=None):
    """A row in the shape the injector and the SwDA parser both emit."""
    record = {"disfluent": disfluent, "clean": clean}
    if annotations is not None:
        record["disfluent_tokens"] = disfluent.split()
        record["annotations"] = annotations
    return record


# --------------------------------------------------------------------------
# normalisation and the baseline
# --------------------------------------------------------------------------


def test_exact_match_ignores_whitespace_and_the_final_mark():
    assert exact_match("Send it.", "Send  it")
    assert exact_match("Send it", "Send it.")


def test_exact_match_is_case_sensitive():
    """Restoring casing is part of the job, so it has to be scored."""
    assert not exact_match("send it.", "Send it.")


def test_copy_baseline_applies_only_the_trivial_rule():
    assert copy_baseline("i sent it to sarah") == "I sent it to sarah."
    assert copy_baseline("um send it") == "Um send it."


def test_word_tokens_strips_punctuation_and_case():
    assert word_tokens("Send it, to Sarah!") == ["send", "it", "to", "sarah"]


# --------------------------------------------------------------------------
# filler removal
# --------------------------------------------------------------------------


def test_filler_removed_scores_a_true_positive():
    r = row(
        "um send the report",
        "Send the report.",
        [{"type": "filler", "interregnum": span(0, 1)}],
    )
    prf = filler_removal_counts(r, "Send the report.")
    assert (prf.tp, prf.fp, prf.fn) == (1, 0, 0)


def test_filler_left_in_scores_a_false_negative():
    r = row(
        "um send the report",
        "Send the report.",
        [{"type": "filler", "interregnum": span(0, 1)}],
    )
    prf = filler_removal_counts(r, "Um send the report.")
    assert (prf.tp, prf.fn) == (0, 1)


def test_deleting_a_literal_filler_word_is_a_false_positive():
    """The characteristic real-speech failure: 'like' as a verb, deleted."""
    r = row("i like italian food", "I like Italian food.", [])
    prf = filler_removal_counts(r, "Italian food.")
    assert prf.fp == 1


def test_one_word_used_both_ways_is_counted_by_occurrence():
    """Deleting the filled pause and keeping the verb is a perfect answer."""
    r = row(
        "i like um like italian food",
        "I like Italian food.",
        [{"type": "filler", "interregnum": span(2, 4)}],
    )
    prf = filler_removal_counts(r, "I like Italian food.")
    assert (prf.tp, prf.fp, prf.fn) == (2, 0, 0)


def test_deletions_owed_to_other_disfluencies_are_not_charged():
    """A false start that swallows 'you' should not cost filler precision."""
    r = row(
        "can you no send the report",
        "Send the report.",
        [{"type": "false_start", "reparandum": span(0, 2), "interregnum": span(2, 3)}],
    )
    prf = filler_removal_counts(r, "Send the report.")
    assert prf.fp == 0


def test_row_annotations_extend_the_scored_vocabulary():
    """SwDA marks fillers the injector never inserts, such as 'oh'."""
    r = row("oh send the report", "Send the report.",
            [{"type": "filler", "interregnum": span(0, 1)}])
    assert "oh" in filler_vocab_for(r)


# --------------------------------------------------------------------------
# repair resolution
# --------------------------------------------------------------------------


REPAIR = [
    {
        "type": "repair",
        "reparandum": span(3, 4),
        "interregnum": span(4, 6),
        "repair": span(6, 7),
    }
]


def test_repair_resolved_when_reparandum_goes_and_repair_stays():
    r = row("send it to john no wait sarah", "Send it to Sarah.", REPAIR)
    assert repair_resolved(r, "Send it to Sarah.") is True


def test_repair_unresolved_when_the_abandoned_word_survives():
    r = row("send it to john no wait sarah", "Send it to Sarah.", REPAIR)
    assert repair_resolved(r, "Send it to John.") is False


def test_repair_unresolved_when_the_correction_is_dropped():
    r = row("send it to john no wait sarah", "Send it to Sarah.", REPAIR)
    assert repair_resolved(r, "Send it to.") is False


def test_repair_word_that_the_gold_also_deletes_is_not_required():
    """Real transcripts nest a repair inside the repair half of a bigger one.

    Requiring every word of the repair span to survive scored a perfect output
    as a failure, so only words that reach the gold are required.
    """
    r = row(
        "i went to the store to the shop",
        "I went to the shop.",
        [{"type": "repair", "reparandum": span(3, 5), "repair": span(5, 7)}],
    )
    assert repair_resolved(r, "I went to the shop.") is True


def test_rows_with_no_repair_are_not_scored():
    r = row("um send the report", "Send the report.",
            [{"type": "filler", "interregnum": span(0, 1)}])
    assert repair_resolved(r, "Send the report.") is None


def test_unannotated_rows_fall_back_to_the_multiset_difference():
    """The hard set carries no spans, so the metric is recovered from the pair."""
    r = row("send it to john no wait sarah", "Send it to Sarah.")
    assert repair_resolved(r, "Send it to Sarah.") is True
    assert repair_resolved(r, "Send it to John.") is False


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def test_score_reports_zero_wer_for_a_perfect_system():
    rows = [row("um send the report", "Send the report.", [])]
    result = score(rows, ["Send the report."])
    assert result["exact_match"] == 1.0
    assert result["wer"] == 0.0
    assert result["n"] == 1


def test_score_leaves_repair_accuracy_undefined_with_no_repairs():
    rows = [row("send the report", "Send the report.", [])]
    result = score(rows, ["Send the report."])
    assert result["n_repairs"] == 0
    assert result["repair_accuracy"] != result["repair_accuracy"]  # nan


# --------------------------------------------------------------------------
# bootstrap intervals
# --------------------------------------------------------------------------


def sample_rows():
    return [
        row("um send the report", "Send the report.",
            [{"type": "filler", "interregnum": span(0, 1)}]),
        row("send it to john no wait sarah", "Send it to Sarah.", REPAIR),
        row("i i think it is late", "I think it is late.",
            [{"type": "repetition", "reparandum": span(0, 1), "repair": span(1, 2)}]),
    ]


def test_aggregate_matches_score():
    """The bootstrap must describe the same quantity as the point estimate."""
    rows = sample_rows()
    hyps = [r["clean"] for r in rows]
    from mend.evaluate import aggregate, per_row_stats

    assert aggregate(per_row_stats(rows, hyps)) == score(rows, hyps)


def test_corpus_wer_is_edits_over_reference_words():
    """Not the mean of per-sentence rates -- that would weight short rows high."""
    rows = [row("a b c d", "A b c d."), row("x", "Y.")]
    hyps = ["A b c d.", "Z."]  # 0 errors in 4 words, then 1 in 1
    assert score(rows, hyps)["wer"] == pytest.approx(1 / 5)


def test_bootstrap_interval_brackets_the_point_estimate():
    rows = sample_rows()
    hyps = [r["clean"] for r in rows]
    result = score(rows, hyps, draws=200, seed=13)
    assert result["exact_match_lo"] <= result["exact_match"] <= result["exact_match_hi"]


def test_bootstrap_is_deterministic_for_a_seed():
    rows = sample_rows()
    hyps = ["Send the report.", "Send it to John.", "I think it is late."]
    a = score(rows, hyps, draws=200, seed=13)
    b = score(rows, hyps, draws=200, seed=13)
    assert a == b


def test_more_data_gives_a_tighter_interval():
    """The whole point of the intervals: they must react to sample size."""
    small = sample_rows()
    large = sample_rows() * 40
    hyps_s = ["Send the report.", "Send it to John.", "I think it is late."]
    a = score(small, hyps_s, draws=400, seed=13)
    b = score(large, hyps_s * 40, draws=400, seed=13)
    width_a = a["exact_match_hi"] - a["exact_match_lo"]
    width_b = b["exact_match_hi"] - b["exact_match_lo"]
    assert width_b < width_a


def test_no_bootstrap_means_no_interval_keys():
    rows = sample_rows()
    result = score(rows, [r["clean"] for r in rows])
    assert not any(k.endswith(("_lo", "_hi")) for k in result)
