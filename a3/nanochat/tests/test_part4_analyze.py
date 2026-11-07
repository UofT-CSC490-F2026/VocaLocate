"""Validate report statistics against the saved paired trials."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.part4_analyze import paired_summary, read_training_log


ROOT = Path(__file__).resolve().parents[2] / "evaluation"


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


def baseline_inputs():
    names = ("passkey-results.json", "passkey-competing-paired.json",
             "part2-baseline-single-record.json", "part2-baseline-four-record-paired.json")
    return [json.loads((ROOT / "results" / name).read_text(encoding="utf-8")) for name in names]


def test_baseline_comparison_reproduces_all_archived_values():
    from scripts.part4_analyze import baseline_summary

    actual = baseline_summary(*baseline_inputs())
    expected = json.loads((ROOT / "results/part2-baseline-comparison.json").read_text(encoding="utf-8"))
    assert actual == expected
    assert len(actual["results"]) == 9
    assert all(r["four_baseline_correct"] > r["four_b_correct"] for r in actual["results"])
    assert sum(r["ci95_high_pp"] < 0 for r in actual["results"]) == 5
    assert actual["results"][6]["b_minus_baseline_pp"] == -28.125


@pytest.mark.parametrize("change", ["seed", "missing_condition", "total"])
def test_baseline_comparison_rejects_inconsistent_results(change):
    from scripts.part4_analyze import baseline_summary

    inputs = baseline_inputs()
    baseline = inputs[-1]["results"]
    if change == "seed":
        baseline[0]["case_results"][0]["seed"] += 1
    elif change == "missing_condition":
        baseline.pop()
    else:
        baseline[0]["correct"] += 1
    with pytest.raises(ValueError):
        baseline_summary(*inputs)


def test_baseline_summary_cli_rebuilds_json_and_csv(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "scripts.part4_analyze", "--include-baseline",
         "--summary-only", "--output-dir", str(tmp_path)],
        cwd=ROOT.parent / "nanochat", capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for name in ("part2-baseline-comparison.json", "part2-baseline-comparison.csv"):
        assert (tmp_path / "results" / name).read_bytes() == (ROOT / "results" / name).read_bytes()
