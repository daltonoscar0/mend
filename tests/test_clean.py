"""Tests for the command line entry point.

Only the parts that do not need a trained checkpoint: input normalisation and
the argument handling. Whether the model produces good text is what
`mend.evaluate` measures.
"""

import io
import sys

import pytest

from mend.clean import main, normalise, read_lines


class Args:
    def __init__(self, text=()):
        self.text = list(text)


# --------------------------------------------------------------------------
# input normalisation
# --------------------------------------------------------------------------


def test_pasted_written_text_is_put_back_into_the_spoken_register():
    assert (
        normalise("Send it to John, no wait, Sarah — um, by Friday.")
        == "send it to john no wait sarah um by friday"
    )


def test_already_spoken_text_is_left_alone():
    text = "um send the the report to john no wait sarah by friday"
    assert normalise(text) == text


def test_internal_apostrophes_and_hyphens_survive():
    assert normalise("Don't forget the like-minded ones.") == "don't forget the like-minded ones"


def test_whitespace_is_collapsed():
    assert normalise("  send   the    report  ") == "send the report"


def test_tokens_with_no_word_content_are_dropped():
    assert normalise("call sarah — please") == "call sarah please"


def test_empty_input_normalises_to_empty():
    assert normalise("") == ""
    assert normalise("   ") == ""


# --------------------------------------------------------------------------
# where the text comes from
# --------------------------------------------------------------------------


def test_positional_arguments_are_joined_into_one_line():
    assert read_lines(Args(["send", "the", "report"])) == ["send the report"]


def test_stdin_is_read_line_by_line(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("first line\n\n  second line  \n"))
    assert read_lines(Args()) == ["first line", "second line"]


# --------------------------------------------------------------------------
# refusals, which must not need torch or a checkpoint
# --------------------------------------------------------------------------


def test_no_input_is_an_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert main([]) == 1
    assert "nothing to clean" in capsys.readouterr().err


def test_missing_model_explains_how_to_train_one(capsys):
    assert main(["--model", "/definitely/not/here", "hello"]) == 1
    err = capsys.readouterr().err
    assert "no model at" in err
    assert "mend.train" in err


@pytest.mark.parametrize("flag", ["--raw", "--show-input"])
def test_flags_are_accepted(flag):
    # still exits 1 on the missing model, but only after parsing cleanly
    assert main([flag, "--model", "/definitely/not/here", "hello"]) == 1
