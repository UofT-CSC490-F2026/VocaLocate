"""Table schemas for every layer of the lake and the warehouse.

IDs line up with the A2 Part One wish-list schemas: `clip_id` plays the role
of sound_id / imitation_id (it is the SHA-256 of the raw file, so the same
file always gets the same ID), `imitator_id` links to imitator profiles and
`query_id` links text hints to imitations.

Where each Part One schema lives (the warehouse has a view per schema with
exactly its column names, see warehouse.py):

    SFX_Folder            library_roots                          v_sfx_folder
    SFX_File              clips + library_files + sfx_metadata   v_sfx_file
    SFX_Metadata          sfx_metadata                           v_sfx_candidates
    Vocal_Imitation       clips + imitation_metadata             v_vocal_imitation
    Retrieval_Query       queries (+ generated from metadata)    v_retrieval_query
    QVIM_Benchmark_Data   pairs of qvim_dev / qvim_eval          v_qvim_benchmark
    Relevance_Judgement   relevance_judgements                   v_relevance_judgement
    Imitator profiles     imitators                              v_imitator_profile
    Search session logs   search_sessions (written by the app)

Columns added after the first release are nullable, so Parquet files written
before them still load (the warehouse reads with union_by_name).
"""

from __future__ import annotations

from collections.abc import Iterable

import pyarrow as pa

# unpaired_imitation: an imitation kept only as a training negative (e.g. a rejected or practice take).
# It shares its target's group key, so it lands in the target's split, but it never forms a training pair.
ROLES = {"reference", "imitation", "library", "unpaired_imitation"}
STATUSES = {"ok", "rejected", "duplicate"}
SPLITS = {"train", "val", "test"}
LIBRARY_STATUSES = {"active", "deleted"}
# pending and indexing are for the app (a root added but not indexed yet, a run in progress);
# the library flow writes completed or failed at the end of each run.
INDEX_STATUSES = {"pending", "indexing", "completed", "failed"}
TOWERS = {"sfx", "imitation", "clap_audio"}
MINING_STRATEGIES = {"retrieval", "metadata", "random"}
MINING_SELECTIONS = {"band", "fallback", "random"}
DEVICE_TYPES = {"computer_mic", "headset_mic", "studio_mic", "other"}
SEARCH_MODES = {"keyword", "reference_audio", "voice", "voice_plus_text"}


def _schema(*cols: tuple[str, pa.DataType, bool]) -> pa.Schema:
    return pa.schema([pa.field(name, dtype, nullable=nullable) for name, dtype, nullable in cols])


# Bronze: one row per raw file we know about. Files are never modified.
RAW_FILES = _schema(
    ("source", pa.string(), False),
    ("source_version", pa.string(), False),
    ("source_key", pa.string(), False),  # path relative to the dataset root
    ("uri", pa.string(), False),  # where the raw bytes live
    ("file_sha256", pa.string(), False),
    ("size_bytes", pa.int64(), False),
    ("role", pa.string(), False),
    ("label", pa.string(), True),
    ("group_key", pa.string(), True),  # the reference sound this file is, or imitates
    ("imitator_id", pa.string(), True),
    ("project_id", pa.string(), True),
    ("licence", pa.string(), True),
    ("extra_json", pa.string(), True),  # source-specific metadata, e.g. ESC-50 fold
    ("ingested_at", pa.timestamp("us", tz="UTC"), False),
    ("run_id", pa.string(), False),
)

# Silver: one row per raw file after cleaning. Clean audio is 32 kHz mono 16-bit WAV.
CLIPS = _schema(
    ("clip_id", pa.string(), False),
    ("source", pa.string(), False),
    ("source_key", pa.string(), False),
    ("role", pa.string(), False),
    ("label", pa.string(), True),
    ("group_key", pa.string(), True),
    ("imitator_id", pa.string(), True),
    ("project_id", pa.string(), True),
    ("licence", pa.string(), True),
    ("status", pa.string(), False),
    ("reject_reason", pa.string(), True),
    ("orig_sample_rate", pa.int32(), True),
    ("orig_channels", pa.int32(), True),
    ("orig_duration_sec", pa.float32(), True),
    ("orig_format", pa.string(), True),  # soundfile's format: WAV, FLAC, MP3, OGG, AIFF, ...
    ("orig_bit_depth", pa.int32(), True),  # from the subtype; null for lossy formats (MP3, Vorbis, Opus)
    # Background noise of the original recording: 10th percentile of 50 ms frame RMS (audio.noise_floor_dbfs).
    ("noise_floor_dbfs", pa.float32(), True),
    ("duration_sec", pa.float32(), True),
    ("peak_dbfs", pa.float32(), True),
    ("rms_dbfs", pa.float32(), True),
    ("truncated", pa.bool_(), False),
    ("clipped", pa.bool_(), False),
    ("clean_path", pa.string(), True),
    ("config_fingerprint", pa.string(), False),
    ("run_id", pa.string(), False),
)

# Silver: which imitation imitates which reference.
PAIRS = _schema(
    ("source", pa.string(), False),
    ("imitation_clip_id", pa.string(), False),
    ("reference_clip_id", pa.string(), False),
    ("group_key", pa.string(), False),
)

# Silver: text hints recorded with an imitation (voice, text and target triples; the Retrieval_Query schema).
QUERIES = _schema(
    ("query_id", pa.string(), False),
    ("source", pa.string(), False),
    ("imitation_clip_id", pa.string(), False),
    ("text_hint", pa.string(), False),
    ("hint_type", pa.string(), True),
    ("target_group_key", pa.string(), False),
    # The target when the group has exactly one clean reference in the query's own source.
    ("target_clip_id", pa.string(), True),
    # Set only when the query has a split of its own (team recordings: val or test); otherwise
    # the query follows its target's split (see v_retrieval_query).
    ("split", pa.string(), True),
)

# Silver: free-text captions written for an SFX clip (e.g. CAPTDURE). Each caption is a
# text-only query whose target is its own clip, for testing the CLAP text branch.
CAPTIONS = _schema(
    ("caption_id", pa.string(), False),
    ("source", pa.string(), False),
    ("clip_id", pa.string(), False),
    ("caption", pa.string(), False),
    ("language", pa.string(), False),
    ("annotator_id", pa.string(), True),
)

# Silver: optional semantic metadata per SFX (the SFX_Metadata wish-list schema).
# Only training-time mining and analysis read it; the deployed search never does.
SFX_METADATA = _schema(
    ("clip_id", pa.string(), False),
    ("source", pa.string(), False),
    ("category", pa.string(), True),
    ("description", pa.string(), True),
    ("tags", pa.list_(pa.string()), True),
    # Clips cut from the same original recording share this ID (e.g. a Freesound ID).
    ("recording_id", pa.string(), True),
)

# Silver: optional recording metadata per imitation (the Vocal_Imitation wish-list schema's
# vocalization_type and device_type; team recordings only, no public dataset records them).
IMITATION_METADATA = _schema(
    ("clip_id", pa.string(), False),
    ("source", pa.string(), False),
    ("vocalization_type", pa.string(), True),  # free text, e.g. "hum", "whistle", "beatbox"
    ("device_type", pa.string(), True),  # computer_mic, headset_mic, studio_mic or other
)

# Silver: imitator profiles, joined to clips by imitator_id.
IMITATORS = _schema(
    ("imitator_id", pa.string(), False),
    ("source", pa.string(), False),
    ("role", pa.string(), True),  # e.g. "sound designer", "student", "crowd worker"
    ("years_experience", pa.int32(), True),
    ("first_language", pa.string(), True),
    ("music_training", pa.bool_(), True),
    ("consent_scope", pa.string(), True),  # what the recordings may be used for
)

# Silver: human relevance ratings of a candidate SFX for a query (the Relevance_Judgement schema).
RELEVANCE_JUDGEMENTS = _schema(
    ("query_id", pa.string(), False),
    ("candidate_clip_id", pa.string(), False),
    ("relevance", pa.int32(), False),  # 0 (not relevant) to 3 (the sound wanted)
    ("rater_id", pa.string(), True),
    ("source", pa.string(), False),
)

# Search session logs from the app (milestone 4). Defined now so the warehouse always has the
# table; the pipeline itself never writes it.
SEARCH_SESSIONS = _schema(
    ("session_id", pa.string(), False),
    ("imitator_id", pa.string(), True),
    ("search_mode", pa.string(), False),  # keyword, reference_audio, voice or voice_plus_text
    ("query_id", pa.string(), True),
    ("library_size", pa.int64(), False),
    ("results_shown", pa.list_(pa.string()), False),  # clip_ids, in ranked order
    ("previewed_sound_ids", pa.list_(pa.string()), False),
    ("chosen_sound_id", pa.string(), True),
    ("time_to_choice_sec", pa.float32(), True),  # null when nothing was chosen
    ("started_at", pa.timestamp("us", tz="UTC"), False),
)

# Gold: cached log-mel features and a fixed-size summary vector per clean clip.
FEATURES = _schema(
    ("clip_id", pa.string(), False),
    ("source", pa.string(), False),
    ("feature_version", pa.string(), False),
    ("feature_path", pa.string(), False),
    ("n_frames", pa.int32(), False),
    ("summary", pa.list_(pa.float32()), False),
)

# Gold: train/val/test assignment. Every clip of a reference sound lands in one split.
SPLITS_TABLE = _schema(
    ("clip_id", pa.string(), False),
    ("source", pa.string(), False),
    ("source_key", pa.string(), False),
    ("group_key", pa.string(), False),
    ("split", pa.string(), False),
    ("strategy", pa.string(), False),
)

# Gold: encoder embeddings, cached per clip and model so re-indexing only embeds new clips.
EMBEDDINGS = _schema(
    ("clip_id", pa.string(), False),
    ("source", pa.string(), False),
    ("model_version", pa.string(), False),
    ("tower", pa.string(), False),  # "sfx" or "imitation" (QVIM towers), or "clap_audio"
    ("vector", pa.list_(pa.float32()), False),
)

# Gold: mined semi-hard negatives for fine-tuning. This is the "mined hard negatives"
# dataset from A2 Part Two: the relevance schema with a model score instead of a human rating.
HARD_NEGATIVES = _schema(
    ("strategy", pa.string(), False),  # retrieval or metadata
    ("model_version", pa.string(), False),  # the model that scored the candidates
    ("imitation_clip_id", pa.string(), False),
    ("positive_clip_id", pa.string(), False),
    ("negative_clip_id", pa.string(), False),
    ("negative_rank", pa.int32(), False),  # rank of the negative in the model's retrieval list
    ("positive_score", pa.float32(), False),
    ("negative_score", pa.float32(), False),
    ("selection", pa.string(), False),  # band, fallback or random
    ("relation", pa.string(), True),  # how metadata relates the negative to the target
    ("run_id", pa.string(), False),
)

# State for incremental indexing of a user's own library.
LIBRARY_FILES = _schema(
    ("library", pa.string(), False),
    ("path", pa.string(), False),
    ("size_bytes", pa.int64(), False),
    ("mtime_ns", pa.int64(), False),
    ("file_sha256", pa.string(), False),
    ("project_id", pa.string(), True),
    ("status", pa.string(), False),
    ("first_seen", pa.timestamp("us", tz="UTC"), False),
    ("last_seen", pa.timestamp("us", tz="UTC"), False),
    ("root_id", pa.string(), True),  # the library_roots folder the file was found under
    ("rel_path", pa.string(), True),  # path relative to that folder
)

# State: one row per folder a library indexes (the SFX_Folder wish-list schema).
LIBRARY_ROOTS = _schema(
    ("root_id", pa.string(), False),  # first 32 hex digits of the SHA-256 of root_path
    ("library", pa.string(), False),
    ("root_path", pa.string(), False),  # absolute
    ("index_status", pa.string(), False),
    ("last_indexed_at", pa.timestamp("us", tz="UTC"), True),  # last completed index
    ("last_run_id", pa.string(), False),
)

# One row per pipeline run, for lineage and monitoring.
PIPELINE_RUNS = _schema(
    ("run_id", pa.string(), False),
    ("flow", pa.string(), False),
    ("source", pa.string(), False),
    ("started_at", pa.timestamp("us", tz="UTC"), False),
    ("finished_at", pa.timestamp("us", tz="UTC"), False),
    ("status", pa.string(), False),
    ("n_input", pa.int64(), False),
    ("n_processed", pa.int64(), False),
    ("n_reused", pa.int64(), False),
    ("n_ok", pa.int64(), False),
    ("n_rejected", pa.int64(), False),
    ("n_duplicate", pa.int64(), False),
    ("config_fingerprint", pa.string(), False),
    ("message", pa.string(), True),
)

TABLES: dict[str, pa.Schema] = {
    "raw_files": RAW_FILES,
    "clips": CLIPS,
    "pairs": PAIRS,
    "queries": QUERIES,
    "sfx_metadata": SFX_METADATA,
    "captions": CAPTIONS,
    "imitation_metadata": IMITATION_METADATA,
    "imitators": IMITATORS,
    "relevance_judgements": RELEVANCE_JUDGEMENTS,
    "search_sessions": SEARCH_SESSIONS,
    "features": FEATURES,
    "splits": SPLITS_TABLE,
    "embeddings": EMBEDDINGS,
    "hard_negatives": HARD_NEGATIVES,
    "library_files": LIBRARY_FILES,
    "library_roots": LIBRARY_ROOTS,
    "pipeline_runs": PIPELINE_RUNS,
}

_ENUMS: dict[str, dict[str, set[str]]] = {
    "raw_files": {"role": ROLES},
    "clips": {"role": ROLES, "status": STATUSES},
    "splits": {"split": SPLITS},
    "queries": {"split": SPLITS},
    "library_files": {"status": LIBRARY_STATUSES},
    "library_roots": {"index_status": INDEX_STATUSES},
    "embeddings": {"tower": TOWERS},
    "hard_negatives": {"strategy": MINING_STRATEGIES, "selection": MINING_SELECTIONS},
    "imitation_metadata": {"device_type": DEVICE_TYPES},
    "search_sessions": {"search_mode": SEARCH_MODES},
}

# Inclusive (min, max) bounds for integer columns.
_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "relevance_judgements": {"relevance": (0, 3)},
}


class SchemaError(ValueError):
    pass


def to_table(name: str, rows: Iterable[dict]) -> pa.Table:
    """Build a table from row dicts and check it against the schema.

    Types are enforced by pyarrow; we also check required columns for nulls and
    enum columns for unknown values and range columns for out-of-range values, which pyarrow does
    not do. A null in a nullable enum or range column is fine.
    """
    schema = TABLES[name]
    rows = list(rows)
    table = pa.Table.from_pylist(rows, schema=schema)
    for f in schema:
        if not f.nullable and table.column(f.name).null_count:
            raise SchemaError(f"{name}.{f.name} has {table.column(f.name).null_count} null values")
    for col, allowed in _ENUMS.get(name, {}).items():
        bad = set(table.column(col).to_pylist()) - allowed - {None}
        if bad:
            raise SchemaError(f"{name}.{col} has unknown values {sorted(bad)}")
    for col, (lo, hi) in _RANGES.get(name, {}).items():
        bad = {v for v in table.column(col).to_pylist() if v is not None and not lo <= v <= hi}
        if bad:
            raise SchemaError(f"{name}.{col} has values outside {lo}..{hi}: {sorted(bad)}")
    return table
