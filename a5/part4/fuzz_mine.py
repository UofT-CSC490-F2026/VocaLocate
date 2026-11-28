"""A5 Part Four: a last check outside the test suite. Call `mine` with random, often invalid input.

Every call must either return negatives (none of them the positive itself, and no other take of
its recording unless the metadata filter is off) or raise one of the ValueErrors `mining.py`
raises on purpose. Any other outcome is a bug.

    PYTHONPATH="pipeline;demo/src" python a5/part4/fuzz_mine.py   # ':' instead of ';' on Linux/macOS
"""

from __future__ import annotations

import collections
import sys
import warnings

import numpy as np

from vocalocate_demo.data import SFXMeta
from vocalocate_demo.mining import MiningConfig, mine

OURS = ("pool_ids must be unique", "one id and one positive", "pool_ids for", "imitation dim", "NaN or inf",
        "must be 2-D", "MiningConfig.")


def one_call(rng: np.random.Generator) -> str:
    n_q, n_pool = int(rng.integers(0, 12)), int(rng.integers(0, 15))
    d_q = d_pool = int(rng.integers(1, 6))
    if rng.random() < 0.05:
        d_pool = d_q + 1
    kind = rng.choice(["normal", "zeros", "huge", "tiny", "nan"])
    q = rng.standard_normal((n_q, d_q)).astype(np.float32)
    pool = rng.standard_normal((n_pool, d_pool)).astype(np.float32)
    if kind == "zeros":
        q[:], pool[:] = 0, 0
    elif kind == "huge":
        q, pool = q * 1e18, pool * 1e18
    elif kind == "tiny":
        q, pool = q * 1e-30, pool * 1e-30
    elif kind == "nan" and q.size:
        q.flat[0] = np.nan
    if rng.random() < 0.2:  # repeated ids
        ids = [f"s{int(i)}" for i in rng.integers(0, max(n_pool, 1) * 2, size=n_pool)]
    else:
        ids = [f"s{i}" for i in range(n_pool)]
    if rng.random() < 0.05 and ids:
        ids = ids[:-1]
    q_ids = [f"q{i}" for i in range(n_q + (rng.random() < 0.05))]
    pos = [ids[int(rng.integers(len(ids)))] if ids and rng.random() < 0.9 else "missing" for _ in range(n_q)]
    meta = {
        c: SFXMeta(rng.choice([None, "a", "b"]),
                   frozenset(rng.choice(["x", "y", "z"], size=int(rng.integers(0, 3)), replace=False)),
                   rng.choice([None, "r1", "r2", c]))
        for c in ids if rng.random() < 0.8
    }
    try:
        cfg = MiningConfig(
            top_k=int(rng.integers(-1, 6)), band=float(rng.choice([-0.1, 0.0, 0.25, 2.0, np.nan])),
            n_neg=int(rng.integers(-1, 5)), dup_quantile=float(rng.choice([-0.1, 0.0, 0.5, 0.99, 1.0, 1.5])),
            min_shared_tags=int(rng.integers(-1, 3)), use_metadata_filter=bool(rng.random() < 0.8),
            seed=int(rng.integers(1000)),
        )
        strategy = str(rng.choice(["retrieval", "metadata", "random"]))
        mined = mine(strategy, q, q_ids, pos, pool, ids, meta, cfg, "v", "r")
    except ValueError as e:
        if any(k in str(e) for k in OURS):
            return "raised a named ValueError"
        raise
    for r in mined.rows:
        assert r["negative_clip_id"] != r["positive_clip_id"], "positive mined as its own negative"
        if r["relation"] == "same_recording":
            assert strategy == "retrieval" and not cfg.use_metadata_filter, "another take mined as a negative"
    return "returned"


def main(n: int = 1000, seed: int = 2026) -> int:
    warnings.filterwarnings("ignore")  # overflow warnings from the huge inputs are expected
    rng = np.random.default_rng(seed)
    outcomes, failures = collections.Counter(), []
    for trial in range(n):
        try:
            outcomes[one_call(rng)] += 1
        except Exception as e:  # anything else is a bug
            failures.append(f"trial {trial}: {type(e).__name__}: {str(e)[:120]}")
    print(f"{n} random calls: {dict(outcomes)}, failures: {len(failures)}")
    print("\n".join(failures[:10]))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
