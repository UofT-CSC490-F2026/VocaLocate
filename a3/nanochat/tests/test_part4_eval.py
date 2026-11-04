"""Check the paired passkey task with the actual tokenizer used for training."""
from pathlib import Path

from nanochat.tokenizer import RustBPETokenizer
from scripts.part4_eval import PasskeyTask


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
