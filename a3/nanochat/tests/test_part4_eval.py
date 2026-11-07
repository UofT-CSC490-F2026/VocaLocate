"""Check the paired passkey task with the actual tokenizer used for training."""
from pathlib import Path

import torch

from nanochat.tokenizer import RustBPETokenizer
from scripts.part4_eval import (
    CompetingPasskeyTask, PasskeyCase, PasskeyTask, prompt_check_cases, score,
)


def test_cases_keep_the_same_code_at_the_requested_token_distances():
    tokenizer = RustBPETokenizer.from_directory(Path(__file__).resolve().parents[2] / "tokenizer")
    task = PasskeyTask(tokenizer)
    short = task.case(length=512, distance=128, seed=42)
    long = task.case(length=2048, distance=1536, seed=42)
    assert short.candidates == long.candidates
    assert short.correct == long.correct
    for case, length, distance in ((short, 512, 128), (long, 2048, 1536)):
        assert len(case.tokens) == length
        assert case.tokens[0] == tokenizer.get_bos_token_id()
        assert case.tokens.count(case.candidates[case.correct]) == 1
        assert case.tokens[-1 - distance] == case.candidates[case.correct]


def test_competing_cases_keep_all_four_codes_and_the_target_distance():
    tokenizer = RustBPETokenizer.from_directory(Path(__file__).resolve().parents[2] / "tokenizer")
    task = CompetingPasskeyTask(tokenizer)
    short = task.case(length=512, distance=128, seed=20261104)
    long = task.case(length=2048, distance=1536, seed=20261104)
    assert short.candidates == long.candidates
    assert short.correct == long.correct
    for case, length, distance in ((short, 512, 128), (long, 2048, 1536)):
        assert len(case.tokens) == length
        assert all(case.tokens.count(code) == 1 for code in case.candidates)
        assert case.tokens[-1 - distance] == case.candidates[case.correct]


def test_score_can_record_predictions_in_case_order():
    def model(ids):
        return torch.nn.functional.one_hot(ids, num_classes=5).float()

    cases = [PasskeyCase([0, token], [1, 2, 3, 4], target)
             for token, target in ((3, 2), (1, 2), (4, 3))]
    predictions = []
    assert score(model, cases, "cpu", 2, predictions=predictions) == 2
    assert predictions == [2, 0, 3]
    assert score(model, cases, "cpu", 2) == 2


def test_prompt_checks_change_only_the_code_and_keep_original_candidates():
    tokenizer = RustBPETokenizer.from_directory(Path(__file__).resolve().parents[2] / "tokenizer")
    case = PasskeyTask(tokenizer).case(length=2048, distance=768, seed=42)
    unknown = tokenizer.encode(" unknown")
    assert len(unknown) == 1
    removed, swapped = prompt_check_cases([case], unknown[0])
    position = 2048 - 1 - 768
    for edited in (removed[0], swapped[0]):
        assert edited.candidates == case.candidates
        assert len(edited.tokens) == len(case.tokens)
        assert [i for i, (a, b) in enumerate(zip(case.tokens, edited.tokens)) if a != b] == [position]
    assert removed[0].tokens[position] == unknown[0]
    assert removed[0].correct == case.correct
    assert unknown[0] not in case.candidates
    assert swapped[0].correct == (case.correct + 1) % 4
    assert swapped[0].tokens[position] == case.candidates[swapped[0].correct]
    assert case.tokens[position] == case.candidates[case.correct]
