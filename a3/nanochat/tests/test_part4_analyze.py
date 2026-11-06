"""Validate report statistics against the saved paired trials."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.part4_analyze import paired_summary, read_training_log


ROOT = Path(__file__).resolve().parents[2] / "part4"


def saved_trials():
    return json.loads((ROOT / "results/passkey-competing-paired.json").read_text())


def test_all_nine_paired_intervals_reproduce_the_archived_results():
    actual = paired_summary(saved_trials())["results"]
    expected = json.loads((ROOT / "results/passkey-paired-summary.json").read_text())["results"]
    assert len(actual) == len(expected) == 9
    for row, archived in zip(actual, expected, strict=True):
        assert row.keys() == archived.keys()
        for key in row:
            assert row[key] == pytest.approx(archived[key])
    assert actual[5]["difference_pp"] == 31.25
    assert actual[5]["b_only"] == 46
    assert actual[5]["a_only"] == 6
    assert sum(row["ci95_pp"][0] > 0 for row in actual) == 6


@pytest.mark.parametrize("field", ["seed", "target_index", "candidate_token_ids"])
def test_unmatched_trials_are_rejected(field):
    data = deepcopy(saved_trials())
    b = next(row for row in data["results"] if row["step"] == 2520)
    case = b["case_results"][0]
    case[field] = list(reversed(case[field])) if field == "candidate_token_ids" else case[field] + 1
    b["correct"] = sum(c["predicted_index"] == c["target_index"] for c in b["case_results"])
    b["accuracy"] = b["correct"] / b["trials"]
    with pytest.raises(ValueError, match="paired"):
        paired_summary(data)


def test_incorrect_saved_totals_are_rejected():
    data = saved_trials()
    data["results"][0]["correct"] += 1
    with pytest.raises(ValueError, match="total"):
        paired_summary(data)


def test_training_curves_come_from_the_original_logs():
    short = read_training_log(ROOT / "logs/d12-context-part4-512.log")
    long = read_training_log(ROOT / "logs/d12-context-part4-2048.log")
    assert len(short["train"]) == len(long["train"]) == 1260
    assert short["validation"][-1] == (1260, 0.960468)
    assert long["validation"][-1] == (2520, 0.845508)
    assert short["train"][-1] == (1259, 3.172498)
    assert long["train"][-1] == (2519, 2.769035)
