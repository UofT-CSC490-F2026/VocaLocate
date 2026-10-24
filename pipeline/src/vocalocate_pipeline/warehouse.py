"""DuckDB warehouse: the query layer over the lake's Parquet tables.

The lake is the source of truth. Each load rebuilds the DuckDB file from the
Parquet tables into a temporary file and swaps it in, so a failed load never
leaves a half-built warehouse behind.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from .config import PipelineConfig
from .lake import Lake
from .schemas import TABLES

VIEWS = {
    # Imitation/reference pairs ready for contrastive training, with their split.
    "v_training_pairs": """
        SELECT p.source, p.group_key, s.split,
               i.clip_id AS imitation_clip_id, i.clean_path AS imitation_path, i.imitator_id,
               r.clip_id AS reference_clip_id, r.clean_path AS reference_path, r.label,
               fi.feature_path AS imitation_feature_path, fr.feature_path AS reference_feature_path
        FROM pairs p
        JOIN clips i ON i.source = p.source AND i.clip_id = p.imitation_clip_id AND i.status = 'ok'
        JOIN clips r ON r.source = p.source AND r.clip_id = p.reference_clip_id AND r.status = 'ok'
        JOIN splits s ON s.source = r.source AND s.clip_id = r.clip_id
        LEFT JOIN features fi ON fi.source = i.source AND fi.clip_id = i.clip_id
        LEFT JOIN features fr ON fr.source = r.source AND fr.clip_id = r.clip_id
    """,
    # Voice, text and target triples. The target can live in any source.
    "v_triples": """
        SELECT q.query_id, q.text_hint, q.hint_type, q.target_group_key,
               i.clip_id AS imitation_clip_id, i.clean_path AS imitation_path, i.imitator_id,
               t.clip_id AS target_clip_id, t.source AS target_source, t.clean_path AS target_path
        FROM queries q
        JOIN clips i ON i.source = q.source AND i.clip_id = q.imitation_clip_id AND i.status = 'ok'
        LEFT JOIN clips t ON t.group_key = q.target_group_key AND t.role IN ('reference', 'library')
                         AND t.status = 'ok'
    """,
    # What the search app indexes: active library files with clean audio and features.
    "v_library_index": """
        SELECT lf.library, lf.path, lf.project_id, c.clip_id, c.duration_sec,
               f.feature_path, f.summary
        FROM library_files lf
        JOIN clips c ON c.source = 'library_' || lf.library AND c.source_key = lf.path AND c.status = 'ok'
        JOIN features f ON f.source = c.source AND f.clip_id = c.clip_id
        WHERE lf.status = 'active'
    """,
    # Candidate SFX for mining and evaluation: clean references and library clips of
    # dataset sources, with their split and any semantic metadata (category falls back
    # to the dataset label).
    "v_sfx_candidates": """
        SELECT c.source, c.clip_id, c.role, c.group_key, s.split, c.clean_path, f.feature_path,
               coalesce(m.category, c.label) AS category, m.tags, m.recording_id, m.description
        FROM clips c
        JOIN splits s ON s.source = c.source AND s.clip_id = c.clip_id
        JOIN features f ON f.source = c.source AND f.clip_id = c.clip_id
        LEFT JOIN sfx_metadata m ON m.source = c.source AND m.clip_id = c.clip_id
        WHERE c.status = 'ok' AND c.role IN ('reference', 'library') AND c.source NOT LIKE 'library_%'
    """,
    # Imitations kept only as training negatives (rejected or practice takes), with the reference
    # they imitate and its metadata. They share that reference's split, so a negative for a train
    # sound never comes from a test sound's imitations.
    "v_unpaired_imitations": """
        SELECT u.source, u.clip_id, u.group_key, s.split, f.feature_path, u.imitator_id,
               r.clip_id AS target_clip_id, coalesce(m.category, r.label) AS category, m.tags
        FROM clips u
        JOIN splits s ON s.source = u.source AND s.clip_id = u.clip_id
        JOIN features f ON f.source = u.source AND f.clip_id = u.clip_id
        JOIN clips r ON r.source = u.source AND r.group_key = u.group_key AND r.role = 'reference'
                    AND r.status = 'ok'
        LEFT JOIN sfx_metadata m ON m.source = r.source AND m.clip_id = r.clip_id
        WHERE u.status = 'ok' AND u.role = 'unpaired_imitation'
    """,
    # Text-only queries (captions) with their target clip, for testing the CLAP text branch.
    "v_captions": """
        SELECT k.caption_id, k.source, k.caption, k.language, k.annotator_id, s.split,
               c.clip_id AS target_clip_id, c.clean_path AS target_path, m.category, m.recording_id
        FROM captions k
        JOIN clips c ON c.source = k.source AND c.clip_id = k.clip_id AND c.status = 'ok'
        JOIN splits s ON s.source = c.source AND s.clip_id = c.clip_id
        LEFT JOIN sfx_metadata m ON m.source = c.source AND m.clip_id = c.clip_id
    """,
    # --- The A2 Part One wish-list schemas, with exactly their column names (NULL where no dataset
    # has the field), then a few of our own columns. clip_id plays sound_id and imitation_id.
    # SFX_Folder: the folders a user's libraries index.
    "v_sfx_folder": """
        SELECT root_id AS folder_id, root_path AS folder_path, index_status, last_indexed_at, library
        FROM library_roots
    """,
    # SFX_File: every clean SFX, from datasets and libraries. file_path is relative to the folder
    # for library files and to the dataset root otherwise; format fields describe the original file.
    "v_sfx_file": """
        SELECT lf.root_id AS folder_id, c.clip_id AS sound_id, coalesce(lf.rel_path, c.source_key) AS file_path,
               c.orig_format AS audio_format, c.orig_sample_rate AS sample_rate, c.orig_bit_depth AS bit_depth,
               c.orig_channels AS channel, c.orig_duration_sec AS duration_sec,
               coalesce(m.category, c.label) AS category, m.description, m.tags, c.source
        FROM clips c
        LEFT JOIN library_files lf ON c.source = 'library_' || lf.library AND lf.path = c.source_key
                                  AND lf.status = 'active'
        LEFT JOIN sfx_metadata m ON m.source = c.source AND m.clip_id = c.clip_id
        WHERE c.status = 'ok' AND c.role IN ('reference', 'library')
    """,
    # Vocal_Imitation: every clean imitation. noise_level_db is the original recording's noise floor (dBFS).
    "v_vocal_imitation": """
        SELECT c.clip_id AS imitation_id, c.clean_path AS imitation_path, c.duration_sec,
               c.noise_floor_dbfs AS noise_level_db, c.source, c.imitator_id
        FROM clips c
        WHERE c.status = 'ok' AND c.role IN ('imitation', 'unpaired_imitation')
    """,
    # Retrieval_Query: recorded queries (an imitation with a text hint) and, for every other paired
    # imitation, a query generated from the target's metadata. text_descriptor is a list of strings:
    # a recorded hint split on commas, each part trimmed and empty parts dropped ("dog bark, loud,
    # metallic" gives ['dog bark', 'loud', 'metallic']); for a generated query, the target's category
    # then its tags, duplicates removed (first one kept). NULL when nothing is left.
    # A target outside the query's source (team recordings) is the one clean SFX its group key names:
    # the group's reference if it has exactly one, else its only clip. A query's split is its own
    # (team: val or test), else its imitation's or target's, which are the same for a pair.
    # Benchmark sources are left to v_qvim_benchmark.
    "v_retrieval_query": """
        WITH candidates AS (
            SELECT group_key, clip_id, source, role,
                   sum(CASE WHEN role = 'reference' THEN 1 ELSE 0 END) OVER (PARTITION BY group_key) AS n_ref,
                   count(*) OVER (PARTITION BY group_key) AS n
            FROM clips WHERE status = 'ok' AND role IN ('reference', 'library') AND group_key IS NOT NULL
        ),
        targets AS (
            SELECT group_key, clip_id, source FROM candidates
            WHERE (n_ref = 1 AND role = 'reference') OR (n_ref = 0 AND n = 1)
        ),
        generated AS (
            SELECT source, imitation_clip_id, reference_clip_id, source AS reference_source FROM pairs
            UNION ALL
            SELECT i.source, i.clip_id, t.clip_id, t.source
            FROM clips i JOIN targets t ON t.group_key = i.group_key AND t.source <> i.source
            WHERE i.status = 'ok' AND i.role = 'imitation'
        ),
        queries_all AS (
            SELECT q.query_id, q.imitation_clip_id AS imitation_id,
                   regexp_extract_all(q.text_hint, '[^,\\s](?:[^,]*[^,\\s])?') AS text_descriptor,
                   coalesce(q.target_clip_id, t.clip_id) AS target_sound_id,
                   CASE WHEN q.target_clip_id IS NOT NULL THEN q.source ELSE t.source END AS target_source,
                   q.split, q.hint_type, 'recorded' AS query_type
            FROM queries q
            LEFT JOIN targets t ON t.group_key = q.target_group_key
            UNION ALL
            SELECT md5('meta:' || g.imitation_clip_id || g.reference_clip_id), g.imitation_clip_id,
                   list_concat([coalesce(m.category, r.label)], m.tags),
                   g.reference_clip_id, g.reference_source, si.split, CAST(NULL AS VARCHAR), 'metadata'
            FROM generated g
            JOIN clips r ON r.source = g.reference_source AND r.clip_id = g.reference_clip_id
            LEFT JOIN sfx_metadata m ON m.source = r.source AND m.clip_id = r.clip_id
            LEFT JOIN splits si ON si.source = g.source AND si.clip_id = g.imitation_clip_id
            WHERE g.source NOT IN ('qvim_dev', 'qvim_eval')
              AND NOT EXISTS (SELECT 1 FROM queries q
                              WHERE q.source = g.source AND q.imitation_clip_id = g.imitation_clip_id)
        ),
        -- Generated queries: drop NULL, blank and repeated terms, keeping each term's first position.
        meta_terms AS (
            SELECT query_id, unnest(text_descriptor) AS term, generate_subscripts(text_descriptor, 1) AS i
            FROM queries_all WHERE query_type = 'metadata'
        ),
        meta_descriptors AS (
            SELECT query_id, array_agg(term ORDER BY i) AS text_descriptor
            FROM (SELECT query_id, term, min(i) AS i FROM meta_terms
                  WHERE term IS NOT NULL AND trim(term) <> '' GROUP BY query_id, term)
            GROUP BY query_id
        )
        SELECT a.query_id, a.imitation_id,
               CAST(CASE WHEN a.query_type = 'metadata' THEN d.text_descriptor
                         WHEN len(a.text_descriptor) > 0 THEN a.text_descriptor END AS VARCHAR[])
                   AS text_descriptor,
               a.target_sound_id, coalesce(a.split, s.split) AS split, a.hint_type, a.query_type
        FROM queries_all a
        LEFT JOIN meta_descriptors d ON d.query_id = a.query_id
        LEFT JOIN splits s ON s.source = a.target_source AND s.clip_id = a.target_sound_id
    """,
    # QVIM_Benchmark_Data: the challenge's DEV and EVAL query/target pairs.
    "v_qvim_benchmark": """
        SELECT imitation_clip_id AS benchmark_query_id, imitation_clip_id AS imitation_id,
               reference_clip_id AS target_sound_id,
               CASE source WHEN 'qvim_dev' THEN 'Development' ELSE 'Evaluation' END AS benchmark_split
        FROM pairs WHERE source IN ('qvim_dev', 'qvim_eval')
    """,
    # Imitator profiles.
    "v_imitator_profile": """
        SELECT imitator_id, role, years_experience, first_language, music_training, consent_scope, source
        FROM imitators
    """,
    # Mined negatives with the metadata of the target and the negative, for analysis.
    "v_hard_negatives": """
        SELECT h.*, pm.category AS positive_category, nm.category AS negative_category,
               nm.recording_id AS negative_recording_id
        FROM hard_negatives h
        LEFT JOIN sfx_metadata pm ON pm.clip_id = h.positive_clip_id
        LEFT JOIN sfx_metadata nm ON nm.clip_id = h.negative_clip_id
    """,
    # Data-quality summary per source.
    "v_clip_quality": """
        SELECT source, role, status, coalesce(reject_reason, '') AS reason,
               count(*) AS n, round(avg(duration_sec), 2) AS avg_duration_sec,
               sum(CASE WHEN truncated THEN 1 ELSE 0 END) AS n_truncated,
               sum(CASE WHEN clipped THEN 1 ELSE 0 END) AS n_clipped
        FROM clips GROUP BY ALL ORDER BY ALL
    """,
    # The same audio file in more than one source (e.g. VimSketch and the Vocal Imitation Set).
    "v_cross_source_duplicates": """
        SELECT clip_id, list(DISTINCT source ORDER BY source) AS sources, count(*) AS n
        FROM clips WHERE status = 'ok'
        GROUP BY clip_id HAVING count(DISTINCT source) > 1
    """,
}


def load(cfg: PipelineConfig) -> Path:
    lake = Lake(cfg.lake_root)
    target = cfg.warehouse_path
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".duckdb.tmp")
    tmp.unlink(missing_ok=True)

    con = duckdb.connect(str(tmp))
    try:
        for name, schema in TABLES.items():
            parts = lake.parts(name)
            if parts:
                files = ", ".join(f"'{p.as_posix()}'" for p in parts)
                con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_parquet([{files}], union_by_name = true)")
            else:
                empty = schema.empty_table()
                con.register("empty_tbl", empty)
                con.execute(f"CREATE TABLE {name} AS SELECT * FROM empty_tbl")
                con.unregister("empty_tbl")
        for name, sql in VIEWS.items():
            con.execute(f"CREATE VIEW {name} AS {sql}")
    finally:
        con.close()
    tmp.replace(target)
    return target


def query(cfg: PipelineConfig, sql: str):
    con = duckdb.connect(str(cfg.warehouse_path), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()
