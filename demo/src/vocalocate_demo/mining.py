"""Candidate-side semi-hard negative mining: the two VocaLocate approaches, plus a random control.

Both approaches keep the same imitation as the anchor and choose a *real SFX*
as the negative (unlike thatsoundslikeme, whose negatives are other vocal
imitations). They differ in where the candidate negatives come from:

Retrieval-driven mining ("learn from real retrieval mistakes")
    1. Score every training-pool SFX for each training imitation with the
       baseline and take the top-K that are not the target.
    2. Filter likely false negatives: other takes of the target's recording
       (metadata) and SFX whose baseline embedding is a near-duplicate of the
       target's, i.e. in the top 1% of SFX-SFX similarities in the pool (audio
       only, so it works without metadata too).
    3. Semi-hard band: keep candidates scored below the positive but within
       `band` of it:  s_pos - band < s_neg < s_pos.  Sample from the band,
       not just the top wrong result.

Pre-constructed metadata-guided mining ("deliberately teach known related SFX")
    1. Before fine-tuning, build a pool of related non-target SFX for each
       target from training metadata, in priority tiers: same category and
       sharing tags, then same category, then sharing tags across categories.
       Other takes of the target's recording are never in the pool.
    2. Score the pool with the baseline and apply the same semi-hard band,
       filling from the highest tier first. These negatives need not be
       current top retrieval mistakes.

Random control: a random non-target SFX from the training pool, so we can tell
whether mining helps beyond simply training for longer (A2 stage 5).

Metadata is only used here, at training time. The deployed search never reads it.

Tests and edge cases (A5 Part 4)
--------------------------------
Why this module. `mine` has cyclomatic complexity 40 (radon), the highest of any algorithmic
function in pipeline/ and demo/: train_recipe is 30, run_library_flow 25, and every other
function is 26 or less. Only two functions score higher, and both are flat branch chains:
cli.main (77) dispatches command-line arguments and summarize (55) formats the report.
recipe.py has the largest module total (115, against 78 here before Part 4), but it is spread
over 13 orchestration functions. Here, 40 decision points sit in one function: three strategies,
a semi-hard band with strict edges, two false-negative filters, three metadata tiers and two
fallbacks, all driven by a seeded RNG. And a wrong negative here does not crash anything. It
quietly trains the model to push away a sound that should match, so the edge cases are mostly
about choosing the *wrong* negative silently.

Each case below is tagged [En] next to the code it guards and in the name of its test in
demo/tests/test_mining_edge_cases.py. The happy path of each strategy is in
demo/tests/test_mining.py. To find each case, we went through every comparison, slice and
fallback in `mine` and asked what happens exactly on the boundary, with nothing to choose from,
and with input that should never occur. We then fed those inputs to the old code: E9-E13 were
real bugs it had, and the rest pin down behaviour that later changes must keep.

Positive tests: the miner picks the right negatives.
  [E1]  Band edges are strict: s_neg == s_pos counts as "above" (too risky), and
        s_neg == s_pos - band is outside the band and only used as a fallback.
        test_e1_band_edges_are_strict_and_the_lower_edge_falls_back. Off-by-one at a float boundary
        is the most likely regression in a band test.
  [E2]  A near-duplicate of the target (the same audio under another id) is never a negative.
        test_e2_near_duplicate_of_the_target_is_filtered. Re-uploads are common in SFX
        libraries, and teaching the model they differ is exactly the false negative we must avoid.
  [E3]  Retrieval-driven mining only looks top_k deep.
        test_e3_retrieval_only_looks_top_k_deep. That is the definition of a "real retrieval
        mistake"; a band candidate at rank 500 is not one.
  [E4]  Metadata tiers fill highest first, even when a lower tier scores closer.
        test_e4_metadata_fills_from_the_highest_tier_first. The tier order is the whole point
        of the metadata-guided approach.
  [E5]  If every related SFX outranks the positive, one is taken at random rather than none.
        test_e5_metadata_takes_one_random_related_sfx_when_all_score_above_the_positive.
        It is the last-resort branch and is easy to break unnoticed.
  [E6]  Each row records the true 1-based rank (positive included) and the exact scores.
        test_e6_rows_record_the_true_rank_and_scores. The mining analysis in the report reads them.
  [E7]  Stats count every query, including ones whose positive is missing from the pool.
        test_e7_stats_count_skipped_queries_and_average_gaps.
  [E8]  An SFX with no metadata row (e.g. a library clip) is "unrelated", not a KeyError.
        test_e8_missing_metadata_is_unrelated_not_an_error.
  [E14] No queries, an empty pool, or a pool holding only the target and its takes: no rows
        and no crash, for all three strategies.
        test_e14_empty_inputs_and_pools_with_nothing_to_mine. Small splits produce these.
  [E15] The same seed gives the same negatives, and different seeds can differ.
        test_e15_the_seed_makes_mining_reproducible. Our multi-seed comparisons depend on it.
  [E16] Two SFX that both lack a category or a recording are not "the same" one, and
        min_shared_tags=0 makes everything share tags.
        test_e16_missing_category_or_recording_never_matches.

Negative tests: bad input fails loudly with a ValueError instead of mining wrong negatives.
  [E9]  Duplicate pool ids: the positive's first copy could be mined as its own negative.
        test_e9_duplicate_pool_ids_are_rejected
  [E10] Ids that do not line up with the embedding rows: zip() dropped extras silently or
        crashed later with an IndexError. test_e10_ids_must_line_up_with_embedding_rows
  [E11] Imitation and SFX embeddings of different sizes (towers from different checkpoints).
        test_e11_embedding_dimensions_must_match
  [E12] NaN or inf embeddings (a diverged checkpoint): retrieval mined nothing, while metadata and
        random mined NaN-scored negatives. test_e12_non_finite_embeddings_are_rejected
  [E13] Out-of-range settings, e.g. n_neg=-1, which returned nothing for two strategies and
        crashed the third. The range edges themselves stay legal.
        test_e13_invalid_settings_fail_at_construction, test_e13_boundary_settings_are_still_valid
  Unknown strategy names: test_mining_rejects_unknown_strategy in test_mining.py.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from .data import SFXMeta


@dataclass
class MiningConfig:
    top_k: int = 20  # retrieval-driven: how deep into the ranked list to look
    band: float = 0.25  # semi-hard band below the positive score
    n_neg: int = 3  # negatives kept per imitation
    # A candidate is a near-duplicate of the target when their SFX-SFX cosine is in the top
    # `dup_quantile` of all pool pairs. Relative, because embedding spaces differ in spread.
    dup_quantile: float = 0.99
    use_metadata_filter: bool = True  # retrieval-driven: drop other takes of the target's recording
    min_shared_tags: int = 2  # metadata-guided: tags needed to count as related across categories
    seed: int = 490

    def __post_init__(self):
        # [E13] A negative n_neg used to return nothing for two strategies and crash the third
        # ("negative dimensions are not allowed"). Out-of-range settings now fail here, at
        # construction, with the setting's name in the message.
        for name in ("top_k", "n_neg", "min_shared_tags"):
            if getattr(self, name) < 0:
                raise ValueError(f"MiningConfig.{name} must be >= 0, got {getattr(self, name)}")
        if not self.band >= 0:  # also rejects NaN
            raise ValueError(f"MiningConfig.band must be >= 0, got {self.band}")
        if not 0.0 <= self.dup_quantile <= 1.0:
            raise ValueError(f"MiningConfig.dup_quantile must be in [0, 1], got {self.dup_quantile}")


def relation(target: SFXMeta, cand: SFXMeta, min_shared_tags: int = 2) -> str:
    # [E16] `is not None`: two clips that both lack a recording or a category are not "the same".
    if target.recording_id is not None and target.recording_id == cand.recording_id:
        return "same_recording"
    same_cat = target.category is not None and target.category == cand.category
    shared = len(target.tags & cand.tags) >= min_shared_tags
    if same_cat and shared:
        return "same_category+tags"
    if same_cat:
        return "same_category"
    if shared:
        return "shared_tags"
    return "unrelated"


METADATA_TIERS = ("same_category+tags", "same_category", "shared_tags")


class _PoolRelations:
    """`relation(pool[p], pool[j])` for every j at once, cached per target p.

    A5 Part One (profiling): `mine` called `relation` once per (imitation, pool SFX) pair,
    4.5 million Python calls for 1,500 imitations over a 1,500-SFX pool, which was over half its
    run time. The relation depends only on the target, so it is computed once per distinct target
    with numpy (categories and recordings as integer codes, tags as a multi-hot matrix) and reused
    by every imitation of that target. The result equals `relation` element for element.
    """

    def __init__(self, pool_meta: list[SFXMeta], min_shared_tags: int):
        from .evaluate import value_codes

        # none_is_unique: a None category or recording never matches (relation checks `is not None`).
        self.cat = value_codes([m.category for m in pool_meta], none_is_unique=True)
        self.rec = value_codes([m.recording_id for m in pool_meta], none_is_unique=True)
        vocab = {t: k for k, t in enumerate(sorted({t for m in pool_meta for t in m.tags}))}
        self.tags = np.zeros((len(pool_meta), max(len(vocab), 1)), dtype=np.int32)
        for i, m in enumerate(pool_meta):
            self.tags[i, [vocab[t] for t in m.tags]] = 1
        self.min_shared_tags = min_shared_tags
        self._cache: dict[int, np.ndarray] = {}

    def of(self, p: int) -> np.ndarray:
        if p not in self._cache:
            same_rec = (self.rec == self.rec[p]) & (self.rec >= 0)
            same_cat = (self.cat == self.cat[p]) & (self.cat >= 0)
            shared = self.tags @ self.tags[p] >= self.min_shared_tags
            rels = np.full(len(self.cat), "unrelated", dtype=object)
            rels[shared] = "shared_tags"
            rels[same_cat] = "same_category"
            rels[same_cat & shared] = "same_category+tags"
            rels[same_rec] = "same_recording"
            self._cache[p] = rels
        return self._cache[p]


@dataclass
class Mined:
    rows: list[dict]
    negatives: dict[str, list[str]]  # imitation clip_id -> negative clip_ids
    stats: dict


def _row(strategy, model_version, qid, pid, nid, rank, s_pos, s_neg, selection, rel, run_id):
    return {
        "strategy": strategy,
        "model_version": model_version,
        "imitation_clip_id": qid,
        "positive_clip_id": pid,
        "negative_clip_id": nid,
        "negative_rank": int(rank),
        "positive_score": float(s_pos),
        "negative_score": float(s_neg),
        "selection": selection,
        "relation": rel,
        "run_id": run_id,
    }


def _finish(strategy: str, rows: list[dict], n_queries: int, counters: Counter, cfg: MiningConfig) -> Mined:
    negatives: dict[str, list[str]] = {}
    for r in rows:
        negatives.setdefault(r["imitation_clip_id"], []).append(r["negative_clip_id"])
    gaps = [r["positive_score"] - r["negative_score"] for r in rows]
    stats = {
        "strategy": strategy,
        "config": asdict(cfg),
        "n_queries": n_queries,
        "n_queries_with_negative": len(negatives),
        "n_negatives": len(rows),
        "selection": dict(Counter(r["selection"] for r in rows)),
        "relation": dict(Counter(r["relation"] for r in rows)),
        "mean_gap": float(np.mean(gaps)) if gaps else None,
        "mean_rank": float(np.mean([r["negative_rank"] for r in rows])) if rows else None,
        **{k: int(v) for k, v in counters.items()},
    }
    return Mined(rows=rows, negatives=negatives, stats=stats)


def _check_inputs(q_emb, q_ids, pos_ids, pool_emb, pool_ids) -> None:
    """Reject inputs that used to crash deep inside `mine` or, worse, silently mine wrong negatives."""
    q_emb, pool_emb = np.asarray(q_emb), np.asarray(pool_emb)
    if q_emb.ndim != 2 or pool_emb.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (rows, dim), got {q_emb.shape} and {pool_emb.shape}")
    # [E10] One imitation id and one positive id per embedding row. zip() used to drop the extras
    # silently, or index past the end of the embeddings with an IndexError.
    if not len(q_ids) == len(pos_ids) == q_emb.shape[0]:
        raise ValueError(
            f"need one id and one positive per imitation row: {q_emb.shape[0]} rows, "
            f"{len(q_ids)} q_ids, {len(pos_ids)} pos_ids"
        )
    if len(pool_ids) != pool_emb.shape[0]:
        raise ValueError(f"{len(pool_ids)} pool_ids for {pool_emb.shape[0]} pool embedding rows")
    # [E11] Both towers must embed into the same space.
    if q_emb.shape[0] and pool_emb.shape[0] and q_emb.shape[1] != pool_emb.shape[1]:
        raise ValueError(f"imitation dim {q_emb.shape[1]} != pool dim {pool_emb.shape[1]}")
    # [E12] NaN or inf (e.g. from a diverged checkpoint) made every comparison False: retrieval
    # quietly mined nothing while metadata and random mined negatives with NaN scores.
    if not (np.isfinite(q_emb).all() and np.isfinite(pool_emb).all()):
        raise ValueError("embeddings contain NaN or inf; is the checkpoint diverged?")
    # [E9] A clip listed twice in the pool: `index` kept the last copy, so the first copy of the
    # positive was not recognised as the positive and could be mined as its own negative.
    if len(set(pool_ids)) != len(pool_ids):
        dupes = sorted(c for c, n in Counter(pool_ids).items() if n > 1)[:5]
        raise ValueError(f"pool_ids must be unique; repeated: {dupes}")


def mine(
    strategy: str,
    q_emb: np.ndarray,
    q_ids: list[str],
    pos_ids: list[str],
    pool_emb: np.ndarray,
    pool_ids: list[str],
    meta: dict[str, SFXMeta],
    cfg: MiningConfig,
    model_version: str,
    run_id: str,
) -> Mined:
    """Choose negatives for each (imitation, positive) pair from the training pool."""
    if strategy not in ("retrieval", "metadata", "random"):
        raise ValueError(f"unknown strategy {strategy!r}")
    _check_inputs(q_emb, q_ids, pos_ids, pool_emb, pool_ids)
    rng = np.random.default_rng(cfg.seed)  # [E15] every random choice below comes from this seed
    index = {cid: i for i, cid in enumerate(pool_ids)}
    # [E8] No metadata row means no category, tags or recording: "unrelated" to everything.
    pool_meta = [meta.get(cid, SFXMeta(None, frozenset(), None)) for cid in pool_ids]
    sims = q_emb @ pool_emb.T
    pool_sims = pool_emb @ pool_emb.T
    off_diag = pool_sims[~np.eye(len(pool_ids), dtype=bool)]
    dup_threshold = float(np.quantile(off_diag, cfg.dup_quantile)) if off_diag.size else 1.0
    order = np.argsort(-sims, axis=1)
    rank_of = np.empty_like(order)  # [E6] 1-based rank of every pool SFX, positive included
    rank_of[np.arange(len(order))[:, None], order] = np.arange(1, order.shape[1] + 1)

    relations = _PoolRelations(pool_meta, cfg.min_shared_tags)
    pool_idx = np.arange(len(pool_ids))

    rows, counters = [], Counter()
    for i, (qid, pid) in enumerate(zip(q_ids, pos_ids)):
        if pid not in index:
            counters["skipped_positive_not_in_pool"] += 1  # [E7], [E14]
            continue
        p = index[pid]
        s_pos = sims[i, p]
        rels = relations.of(p)
        dup = pool_sims[p] > dup_threshold  # [E2] near-duplicates of the target's audio
        not_p = pool_idx != p

        def add(j: int, selection: str):
            rows.append(
                _row(strategy, model_version, qid, pid, pool_ids[j], rank_of[i, j], s_pos, sims[i, j], selection,
                     rels[j], run_id)
            )

        def in_band(j: int) -> bool:
            # [E1] Both edges strict: equal to the positive is too risky, and the lower edge is a fallback.
            return s_pos - cfg.band < sims[i, j] < s_pos

        if strategy == "random":
            choices = np.flatnonzero(not_p & (rels != "same_recording")).tolist()
            for j in rng.choice(choices, size=min(cfg.n_neg, len(choices)), replace=False):
                add(int(j), "random")
            continue

        if strategy == "retrieval":
            # Only the first top_k + 1 entries can matter (p may be one of them); the old version
            # walked the whole ranked row in Python for every imitation.
            top = [int(j) for j in order[i, : cfg.top_k + 1] if j != p][: cfg.top_k]  # [E3]
            kept = []
            for j in top:
                if dup[j]:
                    counters["filtered_near_duplicate"] += 1
                elif cfg.use_metadata_filter and rels[j] == "same_recording":
                    counters["filtered_same_recording"] += 1
                else:
                    kept.append(j)
            counters["above_positive"] += sum(sims[i, j] >= s_pos for j in kept)
            band = [j for j in kept if in_band(j)]
            for j in rng.permutation(band)[: cfg.n_neg]:
                add(int(j), "band")
            if len(band) < cfg.n_neg:
                # Closest-below-positive candidates from the top-K that missed the band.
                below = [j for j in kept if sims[i, j] <= s_pos - cfg.band]
                for j in below[: cfg.n_neg - len(band)]:
                    add(j, "fallback")
            continue

        # Metadata-guided: the pool is fixed by metadata before the model scores anything.
        # Tier membership and the band test are boolean masks over the pool instead of a Python
        # loop over every pool SFX per imitation (same indices, same ascending order).
        usable = not_p & ~dup
        tiers = [np.flatnonzero(usable & (rels == t)) for t in METADATA_TIERS]
        pool = [int(j) for tier in tiers for j in tier]
        counters["metadata_pool_size"] += len(pool)
        if not pool:  # [E8], [E14] nothing related to this target
            counters["empty_metadata_pool"] += 1
            continue
        chosen: list[int] = []
        for tier in tiers:  # [E4] highest tier first, whatever the lower tiers score
            s = sims[i, tier]
            band = tier[(s_pos - cfg.band < s) & (s < s_pos)]
            chosen += [int(j) for j in rng.permutation(band)[: cfg.n_neg - len(chosen)]]
            if len(chosen) == cfg.n_neg:
                break
        for j in chosen:
            add(j, "band")
        if len(chosen) < cfg.n_neg:
            below = sorted((j for j in pool if sims[i, j] <= s_pos - cfg.band), key=lambda j: -sims[i, j])
            fill = below[: cfg.n_neg - len(chosen)]
            for j in fill:
                add(j, "fallback")
            if not chosen and not fill:
                # [E5] Everything related scores above the positive: take one at random rather than nothing.
                add(int(rng.choice(pool)), "random")
    if strategy == "metadata" and q_ids:
        counters["mean_metadata_pool_size"] = counters.pop("metadata_pool_size", 0) // len(q_ids)
    counters["dup_threshold_x1000"] = int(dup_threshold * 1000)
    return _finish(strategy, rows, len(q_ids), counters, cfg)
