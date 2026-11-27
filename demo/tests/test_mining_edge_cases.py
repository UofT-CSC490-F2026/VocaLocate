"""A5 Part Four: edge cases, failure modes and key use cases for `vocalocate_demo.mining`.

Each test is tagged with the edge-case id ([E1] ... [E16]) used in the "Tests and edge cases"
section of mining.py, which explains why each case was chosen. Positive tests check that the
miner picks the right negatives; negative tests check that bad input fails loudly with a
ValueError instead of mining wrong negatives. The happy-path tests for the three strategies
are in test_mining.py.

Scores are built to be exact: the query is [1, 0] and each pool SFX is [s, y], so the
query-SFX similarity is exactly s. That lets the tests sit right on the band boundaries.
"""

from __future__ import annotations

import numpy as np
import pytest

from vocalocate_demo.data import SFXMeta
from vocalocate_demo.mining import MiningConfig, mine, relation

Q = np.array([[1.0, 0.0]], dtype=np.float32)


def _pool(scores: list[float]) -> np.ndarray:
    """Pool embeddings whose similarity to Q is exactly `scores` (all dyadic, so exact in float32)."""
    return np.array([[s, 0.01 * (i + 1)] for i, s in enumerate(scores)], dtype=np.float32)


def _meta(ids, cats=None, recs=None, tags=None) -> dict[str, SFXMeta]:
    n = len(ids)
    cats, recs, tags = cats or ["c"] * n, recs or [f"r{i}" for i in range(n)], tags or [frozenset()] * n
    return {cid: SFXMeta(c, t, r) for cid, c, r, t in zip(ids, cats, recs, tags)}


def _cfg(**kw) -> MiningConfig:
    # dup_quantile=1 switches the near-duplicate filter off unless a test turns it on.
    return MiningConfig(**{"band": 0.25, "n_neg": 2, "dup_quantile": 1.0, **kw})


def _run(strategy, scores, meta=None, cfg=None, ids=None):
    ids = ids or ["pos"] + [f"s{i}" for i in range(1, len(scores))]
    meta = meta if meta is not None else _meta(ids)
    return mine(strategy, Q, ["q"], ["pos"], _pool(scores), ids, meta, cfg or _cfg(), "v", "r")


# ----------------------------------------------------------------------------- positive tests


def test_e1_band_edges_are_strict_and_the_lower_edge_falls_back():
    """[E1] s_neg == s_pos is "above", s_neg == s_pos - band is outside the band (fallback)."""
    ids = ["pos", "equal", "lower_edge", "inside"]
    mined = _run("retrieval", [0.75, 0.75, 0.5, 0.625], ids=ids)
    picked = {r["negative_clip_id"]: r["selection"] for r in mined.rows}
    assert picked == {"inside": "band", "lower_edge": "fallback"}
    assert mined.stats["above_positive"] == 1  # "equal" is too close to be a safe negative


def test_e2_near_duplicate_of_the_target_is_filtered():
    """[E2] An SFX whose embedding is the positive's (a re-upload) must never be a negative."""
    ids = ["pos", "copy", "s2", "s3", "s4"]
    pool = _pool([0.75, 0.75, 0.625, 0.5, 0.25])
    pool[1] = pool[0]  # identical audio under another id
    mined = mine("retrieval", Q, ["q"], ["pos"], pool, ids, _meta(ids), _cfg(dup_quantile=0.9, band=0.5), "v", "r")
    assert "copy" not in mined.negatives["q"]
    assert mined.stats["filtered_near_duplicate"] >= 1


def test_e3_retrieval_only_looks_top_k_deep():
    """[E3] A band candidate ranked below top_k is not a "real retrieval mistake" and is skipped."""
    scores = [0.75, 0.875, 0.625]  # pos, one SFX above it (rank 1), one in the band (rank 3)
    assert _run("retrieval", scores, cfg=_cfg(top_k=1)).rows == []
    assert _run("retrieval", scores, cfg=_cfg(top_k=2)).negatives["q"] == ["s2"]


def test_e4_metadata_fills_from_the_highest_tier_first():
    """[E4] Same category + tags beats same category, even when the lower tier scores closer."""
    ids = ["pos", "cat_only", "cat_and_tags"]
    tags = [frozenset({"a", "b"}), frozenset(), frozenset({"a", "b"})]
    meta = _meta(ids, tags=tags)
    mined = _run("metadata", [0.75, 0.70, 0.55], meta=meta, ids=ids, cfg=_cfg(n_neg=1))
    assert mined.negatives["q"] == ["cat_and_tags"]
    assert mined.rows[0]["relation"] == "same_category+tags"


def test_e5_metadata_takes_one_random_related_sfx_when_all_score_above_the_positive():
    """[E5] Everything related outranks the positive: one random pick beats no negative at all."""
    mined = _run("metadata", [0.5, 0.75, 0.875])
    assert len(mined.rows) == 1
    assert mined.rows[0]["selection"] == "random"
    assert mined.rows[0]["negative_clip_id"] in {"s1", "s2"}


def test_e6_rows_record_the_true_rank_and_scores():
    """[E6] negative_rank is the 1-based rank in the full list, positive included (used in analysis)."""
    mined = _run("retrieval", [0.75, 0.875, 0.625], cfg=_cfg(n_neg=1))
    (row,) = mined.rows
    assert (row["negative_clip_id"], row["negative_rank"]) == ("s2", 3)
    assert (row["positive_score"], row["negative_score"]) == (0.75, 0.625)


def test_e7_stats_count_skipped_queries_and_average_gaps():
    """[E7] Run statistics are stored with every mining run; they must count every query."""
    ids = ["pos", "s1", "s2"]
    q = np.vstack([Q, Q])
    mined = mine("retrieval", q, ["q1", "q2"], ["pos", "not_in_pool"], _pool([0.75, 0.625, 0.5]), ids,
                 _meta(ids), _cfg(n_neg=1), "v", "r")
    s = mined.stats
    assert (s["n_queries"], s["n_queries_with_negative"], s["skipped_positive_not_in_pool"]) == (2, 1, 1)
    assert s["mean_gap"] == pytest.approx(0.125)


def test_e8_missing_metadata_is_unrelated_not_an_error():
    """[E8] A pool SFX with no metadata row (e.g. a library clip) is "unrelated", never a crash."""
    ids = ["pos", "s1", "s2"]
    meta = {"pos": SFXMeta("c", frozenset(), "r0")}  # s1, s2 have no metadata
    assert _run("metadata", [0.75, 0.625, 0.5], meta=meta).stats["empty_metadata_pool"] == 1
    mined = _run("retrieval", [0.75, 0.625, 0.5], meta=meta)
    assert {r["relation"] for r in mined.rows} == {"unrelated"}


@pytest.mark.parametrize("strategy", ["retrieval", "metadata", "random"])
def test_e14_empty_inputs_and_pools_with_nothing_to_mine(strategy):
    """[E14] No queries, an empty pool, or a pool of only the target and its takes: no rows, no crash."""
    cfg = _cfg()
    empty = mine(strategy, Q[:0], [], [], _pool([0.5]), ["pos"], {}, cfg, "v", "r")
    assert empty.rows == [] and empty.stats["mean_gap"] is None
    no_pool = mine(strategy, Q, ["q"], ["pos"], np.zeros((0, 2), np.float32), [], {}, cfg, "v", "r")
    assert no_pool.stats["skipped_positive_not_in_pool"] == 1
    ids = ["pos", "take"]
    takes = _meta(ids, recs=["r0", "r0"])
    assert _run(strategy, [0.75, 0.625], meta=takes, ids=ids).rows == []


def test_e15_the_seed_makes_mining_reproducible():
    """[E15] Same seed, same negatives (reruns are comparable); another seed can change random picks."""
    scores = [0.75] + [0.5] * 40
    a, b = _run("random", scores, cfg=_cfg(seed=1)), _run("random", scores, cfg=_cfg(seed=1))
    assert a.rows == b.rows
    others = {tuple(_run("random", scores, cfg=_cfg(seed=s)).negatives["q"]) for s in range(2, 7)}
    assert len(others) > 1


def test_e16_missing_category_or_recording_never_matches():
    """[E16] Two SFX that both lack a category or recording are not "the same" one."""
    blank = SFXMeta(None, frozenset(), None)
    assert relation(blank, blank) == "unrelated"
    assert relation(blank, blank, min_shared_tags=0) == "shared_tags"  # 0 shared tags needed: everything shares


# ----------------------------------------------------------------------------- negative tests


def test_e9_duplicate_pool_ids_are_rejected():
    """[E9] A repeated id used to let the positive's first copy be mined as its own negative."""
    ids = ["pos", "s1", "pos"]
    with pytest.raises(ValueError, match=r"pool_ids must be unique; repeated: \['pos'\]"):
        _run("random", [0.75, 0.5, 0.75], ids=ids)


@pytest.mark.parametrize(
    "q_ids,pos_ids,pool_ids",
    [(["q", "extra"], ["pos"], ["pos", "s1"]), (["q"], ["pos", "extra"], ["pos", "s1"]), (["q"], ["pos"], ["pos"])],
    ids=["extra_query_id", "extra_positive", "pool_ids_short"],
)
def test_e10_ids_must_line_up_with_embedding_rows(q_ids, pos_ids, pool_ids):
    """[E10] Mismatched ids used to be dropped by zip() or crash later with an IndexError."""
    with pytest.raises(ValueError, match="pool_ids for|one id and one positive"):
        mine("retrieval", Q, q_ids, pos_ids, _pool([0.75, 0.5]), pool_ids, {}, _cfg(), "v", "r")


def test_e11_embedding_dimensions_must_match():
    """[E11] Imitation and SFX towers from different checkpoints: a clear error, not a matmul error."""
    with pytest.raises(ValueError, match="imitation dim 3 != pool dim 2"):
        mine("retrieval", np.ones((1, 3), np.float32), ["q"], ["pos"], _pool([0.75]), ["pos"], {}, _cfg(), "v", "r")
    # A single query passed as a 1-D vector instead of a (1, dim) matrix.
    with pytest.raises(ValueError, match="must be 2-D"):
        mine("retrieval", Q[0], ["q"], ["pos"], _pool([0.75]), ["pos"], {}, _cfg(), "v", "r")


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("where", ["query", "pool"])
def test_e12_non_finite_embeddings_are_rejected(bad, where):
    """[E12] NaN/inf used to mine nothing (retrieval) or NaN-scored negatives (metadata, random)."""
    q, pool = Q.copy(), _pool([0.75, 0.5])
    (q if where == "query" else pool)[0, 0] = bad
    with pytest.raises(ValueError, match="NaN or inf"):
        mine("metadata", q, ["q"], ["pos"], pool, ["pos", "s1"], {}, _cfg(), "v", "r")


@pytest.mark.parametrize(
    "field,value",
    [("n_neg", -1), ("top_k", -1), ("min_shared_tags", -1), ("band", -0.1), ("band", float("nan")),
     ("dup_quantile", 1.5), ("dup_quantile", -0.01)],
)
def test_e13_invalid_settings_fail_at_construction(field, value):
    """[E13] A negative n_neg used to return nothing for two strategies and crash the third."""
    with pytest.raises(ValueError, match=f"MiningConfig.{field}"):
        MiningConfig(**{field: value})


def test_e13_boundary_settings_are_still_valid():
    """[E13] The edges of each range are legal: no negatives, no depth, keep every pair, no filter."""
    cfg = MiningConfig(n_neg=0, top_k=0, band=0.0, dup_quantile=0.0, min_shared_tags=0)
    assert _run("retrieval", [0.75, 0.5], cfg=cfg).rows == []
