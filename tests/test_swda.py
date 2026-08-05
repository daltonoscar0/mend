"""Tests for the SwDA disfluency-markup parser."""

import pytest

from mend.swda import parse_utterance, split_by_conversation, tidy_clean


def parse(text):
    result = parse_utterance(text)
    assert result is not None, text
    return result


def clean_of(text):
    return " ".join(tidy_clean(parse(text).clean_tokens))


def disfluent_of(text):
    return " ".join(parse(text).disfluent_tokens)


def annotations_of(text, kind=None):
    anns = parse(text).annotations
    return [a for a in anns if kind is None or a.type == kind]


def span_words(result, span):
    return " ".join(result.disfluent_tokens[span.start : span.end])


# --------------------------------------------------------------------------
# what survives to the clean side
# --------------------------------------------------------------------------


def test_reparandum_is_dropped_and_repair_is_kept():
    text = "[ has rejected, + they've rejected ] the answer. /"
    assert clean_of(text) == "They've rejected the answer."
    assert disfluent_of(text) == "has rejected they've rejected the answer"


def test_filled_pause_is_dropped():
    assert clean_of("{F Uh, } I think so. /") == "I think so."


def test_discourse_marker_is_dropped():
    assert clean_of("{D You know, } it works. /") == "It works."


def test_editing_term_is_dropped():
    assert clean_of("{E I mean, } they reimburse everything /") == "They reimburse everything."


def test_coordinating_conjunction_is_kept():
    assert clean_of("{C And } we went home. /") == "And we went home."


def test_nonverbal_markers_are_stripped():
    assert clean_of("We went <laughter> home. /") == "We went home."
    assert clean_of("We went ((home)). /") == "We went home."


# --------------------------------------------------------------------------
# annotation structure
# --------------------------------------------------------------------------


def test_repeated_words_are_typed_as_a_repetition():
    text = "[ do you, + do you ] have any kids /"
    anns = annotations_of(text)
    assert [a.type for a in anns] == ["repetition"]
    result = parse(text)
    assert span_words(result, anns[0].reparandum) == "do you"
    assert span_words(result, anns[0].repair) == "do you"


def test_substitution_is_typed_as_a_repair():
    text = "[ has rejected, + they've rejected ] the answer /"
    anns = annotations_of(text)
    assert [a.type for a in anns] == ["repair"]
    result = parse(text)
    assert span_words(result, anns[0].reparandum) == "has rejected"
    assert span_words(result, anns[0].repair) == "they've rejected"


def test_editing_term_between_the_halves_becomes_the_interregnum():
    text = "[ the report, + {E I mean, } the summary ] is late /"
    result = parse(text)
    repair = [a for a in result.annotations if a.type == "repair"][0]
    assert span_words(result, repair.reparandum) == "the report"
    assert span_words(result, repair.interregnum) == "i mean"
    assert span_words(result, repair.repair) == "the summary"
    assert repair.boundary_index() == repair.interregnum.start


def test_abandoned_repair_with_no_correction_is_a_false_start():
    text = "{C But } I used to work [ and, + ] when I had two children /"
    anns = annotations_of(text, "false_start")
    assert len(anns) == 1
    assert anns[0].repair is None
    assert clean_of(text) == "But I used to work when I had two children."


def test_standalone_filled_pause_becomes_a_filler_annotation():
    text = "We went, {F uh, } home for the weekend /"
    anns = annotations_of(text, "filler")
    assert len(anns) == 1
    result = parse(text)
    assert span_words(result, anns[0].interregnum) == "uh"


def test_nested_repair_keeps_only_the_outer_repair():
    text = "[ [ I, + {F uh, } my sister has a, ] + she just had a ] baby /"
    assert clean_of(text) == "She just had a baby."


# --------------------------------------------------------------------------
# internal consistency, the property the evaluation relies on
# --------------------------------------------------------------------------


CASES = [
    "[ has rejected, + they've rejected ] the answer. /",
    "[ do you, + do you ] have any kids /",
    "{F Uh, } I think so. /",
    "{C And } we went home. /",
    "[ the report, + {E I mean, } the summary ] is late /",
    "[ [ I, + {F uh, } my sister has a, ] + she just had a ] baby /",
    "{C But } I used to work [ and, + ] when I had two children /",
    "[ {C and } you, + {C and } you ] don't always even know /",
]


@pytest.mark.parametrize("text", CASES)
def test_alignment_maps_kept_tokens_onto_the_clean_side(text):
    result = parse(text)
    sources = [s for s in result.alignment if s is not None]
    assert sources == list(range(len(result.clean_tokens)))
    assert len(result.alignment) == len(result.disfluent_tokens)


@pytest.mark.parametrize("text", CASES)
def test_spans_stay_inside_the_token_sequence(text):
    result = parse(text)
    n = len(result.disfluent_tokens)
    for ann in result.annotations:
        for span in (ann.reparandum, ann.interregnum, ann.repair):
            if span is not None:
                assert 0 <= span.start < span.end <= n


@pytest.mark.parametrize("text", CASES)
def test_disfluent_side_is_lowercase_and_unpunctuated(text):
    disfluent = disfluent_of(text)
    assert disfluent == disfluent.lower()
    assert not any(c in disfluent for c in '.,;:!?"')


# --------------------------------------------------------------------------
# what gets refused
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I was going to say, -/",          # continues into the next utterance
        "-- and then we left. /",          # continued from the previous one
        "[ I don't, + I don't have any. /",  # unbalanced bracket
        "{F uh, I think so. /",            # unbalanced brace
        "no slash unit here",              # not a complete unit
    ],
)
def test_incomplete_or_unbalanced_utterances_are_refused(text):
    assert parse_utterance(text) is None


def test_tidy_clean_capitalises_and_closes_the_sentence():
    assert tidy_clean([",", "we", "went", "home"]) == ["We", "went", "home."]
    assert tidy_clean(["We", "went", "home?"]) == ["We", "went", "home?"]
    assert tidy_clean([]) == []


# --------------------------------------------------------------------------
# the conversation-disjoint split
# --------------------------------------------------------------------------


def _rows(n_conversations=20, per_conversation=10):
    return [
        {
            "disfluent": f"utterance {c} {i}",
            "clean": f"Utterance {c} {i}.",
            "annotations": [{"type": "repair"}] if i % 5 == 0 else [],
            "conversation": str(c),
        }
        for c in range(n_conversations)
        for i in range(per_conversation)
    ]


def test_split_keeps_conversations_whole():
    """No conversation may appear in two splits, or the held-out set leaks."""
    splits = split_by_conversation(_rows(), seed=13)
    seen = {}
    for name, rows in splits.items():
        for row in rows:
            other = seen.setdefault(row["conversation"], name)
            assert other == name, (
                f"conversation {row['conversation']} is in both {other} and {name}"
            )


def test_split_loses_no_utterances():
    rows = _rows()
    splits = split_by_conversation(rows, seed=13)
    assert sum(len(v) for v in splits.values()) == len(rows)


def test_split_is_deterministic_for_a_seed():
    a = split_by_conversation(_rows(), seed=7)
    b = split_by_conversation(_rows(), seed=7)
    c = split_by_conversation(_rows(), seed=8)
    assert [r["disfluent"] for r in a["eval"]] == [r["disfluent"] for r in b["eval"]]
    assert [r["disfluent"] for r in a["eval"]] != [r["disfluent"] for r in c["eval"]]


def test_split_respects_the_requested_fractions():
    rows = _rows(n_conversations=100, per_conversation=10)
    splits = split_by_conversation(rows, seed=13, val_fraction=0.1, eval_fraction=0.3)
    # whole conversations only, so allow one conversation of slack either way
    assert 0.3 <= len(splits["eval"]) / len(rows) <= 0.32
    assert 0.1 <= len(splits["val"]) / len(rows) <= 0.12


def test_every_split_carries_repairs():
    """A held-out set with no repairs in it cannot measure repair resolution."""
    splits = split_by_conversation(_rows(), seed=13)
    for name, rows in splits.items():
        repairs = sum(1 for r in rows for a in r["annotations"] if a["type"] == "repair")
        assert repairs > 0, f"{name} has no repairs to score"
