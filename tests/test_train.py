"""Tests for the data side of training.

Only the mixing logic is covered here. Anything needing a checkpoint belongs in
the evaluation, so that ``pytest`` still runs on a fresh clone with no model.
"""

import json

import pytest

from mend.train import load_pairs, mix_real_speech


@pytest.fixture
def real_file(tmp_path):
    path = tmp_path / "swda_train.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i in range(100):
            fh.write(
                json.dumps(
                    {
                        "disfluent": f"real utterance {i}",
                        "clean": f"Real utterance {i}.",
                        "annotations": [],
                        "conversation": str(i // 10),
                    }
                )
                + "\n"
            )
    return path


def synthetic(n=50):
    return [{"disfluent": f"synth {i}", "clean": f"Synth {i}."} for i in range(n)]


def test_mixing_is_a_no_op_when_disabled(real_file):
    rows = synthetic()
    mixed, n_real = mix_real_speech(rows, real_file, limit=0, seed=13)
    assert n_real == 0
    assert mixed == rows


def test_missing_file_degrades_to_synthetic_only(tmp_path):
    rows = synthetic()
    mixed, n_real = mix_real_speech(rows, tmp_path / "absent.jsonl", limit=10, seed=13)
    assert n_real == 0
    assert mixed == rows


def test_cap_limits_the_real_share(real_file):
    mixed, n_real = mix_real_speech(synthetic(50), real_file, limit=25, seed=13)
    assert n_real == 25
    assert len(mixed) == 75
    assert sum(r["disfluent"].startswith("real") for r in mixed) == 25


def test_negative_limit_takes_everything(real_file):
    mixed, n_real = mix_real_speech(synthetic(50), real_file, limit=-1, seed=13)
    assert n_real == 100
    assert len(mixed) == 150


def test_real_rows_are_interleaved_not_appended(real_file):
    """A block of real rows at the tail would make the last epoch lopsided."""
    mixed, _ = mix_real_speech(synthetic(50), real_file, limit=50, seed=13)
    kinds = [r["disfluent"].startswith("real") for r in mixed]
    assert any(kinds[:25]) and not all(kinds[:25])


def test_mixing_is_deterministic_for_a_seed(real_file):
    a, _ = mix_real_speech(synthetic(), real_file, limit=25, seed=13)
    b, _ = mix_real_speech(synthetic(), real_file, limit=25, seed=13)
    c, _ = mix_real_speech(synthetic(), real_file, limit=25, seed=14)
    assert a == b
    assert a != c


def test_only_the_two_training_fields_are_kept(real_file):
    """The span annotations are for evaluation; carrying them wastes memory."""
    mixed, _ = mix_real_speech(synthetic(1), real_file, limit=5, seed=13)
    for row in mixed:
        assert set(row) == {"disfluent", "clean"}


def test_load_pairs_respects_its_limit(real_file):
    assert len(load_pairs(real_file, limit=7)) == 7
    assert len(load_pairs(real_file)) == 100
