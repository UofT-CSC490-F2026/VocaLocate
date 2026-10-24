# VocaLocate data pipeline

Initial version of the A2 Part Three data-processing pipeline. It ingests the
datasets from A2 Part Two and a user's own SFX folders, cleans the audio,
computes features and train/val/test splits, and loads everything into a
DuckDB warehouse.

The same code runs on a laptop against a local `lake/` folder, or on AWS against
an S3 lake (ECS Fargate tasks started by Step Functions, all defined in Terraform
in [`../infra`](../infra)).

![Pipeline architecture](docs/diagrams/architecture.png)

## Layers

| Layer | Contents | Format |
| --- | --- | --- |
| Bronze | Raw files exactly as downloaded, plus a `raw_files` manifest (hash, size, labels, licence) | original audio + Parquet |
| Silver | Clean audio (mono, 32 kHz, trimmed, loudness-normalised) and the `clips`, `pairs`, `queries`, `sfx_metadata`, `captions` and `imitators` tables | 16-bit WAV + Parquet |
| Gold | Log-mel features with the QVIM baseline settings, a 256-value summary vector per clip, and `splits` | float16 `.npy` + Parquet |
| State | `library_files` (one row per file seen) and `library_roots` (one row per indexed folder) per library | Parquet |
| Warehouse | All tables plus views for training, triples, the library index, data quality and each A2 Part One schema | DuckDB |

Table schemas are in [`vocalocate_pipeline/schemas.py`](vocalocate_pipeline/schemas.py),
and the folder layout is in [`lake.py`](vocalocate_pipeline/lake.py).

## Two flows

* **Dataset flow** (`ingest-dataset`): VimSketch, the QVIM DEV set, ESC-50,
  team recordings and `sfx_meta` (any SFX set with a category/tags/recording
  metadata CSV). Run when a dataset is first downloaded, when its version
  changes, or when the cleaning/feature settings change.
* **Library flow** (`index-library`): a user's SFX folders, processed
  incrementally. Unchanged files are skipped by size and modification time,
  moved files are recognised by their hash, and deleted files are marked as
  deleted. See [`docs/diagrams/library_flow.png`](docs/diagrams/library_flow.png).

Both flows cache work by content hash and config fingerprint, so a re-run only
processes new or changed files. Every run writes a row to `pipeline_runs`.

## Part One schemas

Every A2 Part One wish-list schema has a warehouse view with exactly its column
names, in order, followed by a few of our own columns. `clip_id` (the SHA-256
of the raw file) plays `sound_id` and `imitation_id`. A field no dataset
provides is NULL.

| Part One schema | View | Built from |
| --- | --- | --- |
| SFX_Folder | `v_sfx_folder` | `library_roots`: `root_id` (hash of the absolute path), status `pending` / `indexing` / `completed` / `failed`, last completed index |
| SFX_File | `v_sfx_file` | ok reference and library `clips` (original `orig_format`, `orig_bit_depth`, sample rate, channels, duration), `library_files.rel_path` and `root_id`, `sfx_metadata` |
| Vocal_Imitation | `v_vocal_imitation` | ok imitation `clips`; `noise_level_db` is `noise_floor_dbfs`, the 10th percentile of 50 ms frame RMS of the original recording |
| Retrieval_Query | `v_retrieval_query` | recorded `queries` (`query_type` `recorded`) plus one query per other paired or team imitation, its text the target's category and tags (`metadata`); QVIM sets excluded. `text_descriptor` is a list of strings, see below |
| QVIM_Benchmark_Data | `v_qvim_benchmark` | `pairs` of `qvim_dev` (Development) and `qvim_eval` (Evaluation) |
| Relevance_Judgement | not stored | no dataset we use has human relevance ratings |
| Imitator profiles | `v_imitator_profile` | `imitators` |
| Search session logs | not stored | nothing logs searches yet |

Vocal_Imitation's `vocalization_type` and `device_type` are left out: no dataset
we use records them.

`text_descriptor` (`VARCHAR[]`) is the text as the backend uses it, a list of
terms. For a recorded query it is the hint split on commas, each part trimmed and
empty parts dropped: `"dog bark, loud,  metallic"` gives `['dog bark', 'loud',
'metallic']`, and a sentence with no commas stays one element. For a generated
query it is the target's category followed by its tags, repeats removed (the first
one kept): category `bell` and tags `metallic; bell; ringing` give `['bell',
'metallic', 'ringing']`. It is NULL when nothing is left. The raw hint, as typed,
stays in `queries.text_hint` (and `v_triples.text_hint`).

Columns added since the first release (`orig_format`, `orig_bit_depth`,
`noise_floor_dbfs`, `queries.target_clip_id` and `split`,
`library_files.root_id` and `rel_path`) are nullable: older Parquet files still
load, and clips reused from the cache of an older run keep NULLs until they are
cleaned again (e.g. after a settings change).

### Team recordings

`recordings.csv` needs `file, imitator_id, target_group_key, text_hint,
hint_type` and may add `split`: `val` or `test`. Team recordings are never
trained on; Part Two splits the target sounds before recording into a validation
group (for tuning the fusion weight) and a test group. All recordings of a target
share its split, and a target with none gets a seeded 50/50 val/test hash of its
group key.

An optional `imitators.csv` next to it holds `imitator_id, role,
years_experience, first_language, music_training, consent_scope`.

The Vocal Imitation Set's and VocalSketch's `participant_survey*.csv` also fill
`imitators` (role, years of music-technology use, and whether the participant
makes music); neither dataset records the first language or consent terms.

## Usage

```bash
cd pipeline
pip install -e '.[dev]'

# ESC-50: git clone https://github.com/karolpiczak/ESC-50
vocalocate-pipeline --lake lake --workers 4 -v ingest-dataset --source esc50 --path ESC-50
vocalocate-pipeline --lake lake ingest-dataset --source vimsketch --path Vim_Sketch_Dataset
vocalocate-pipeline --lake lake ingest-dataset --source qvim_dev --path qvim-dev
vocalocate-pipeline --lake lake index-library --root ~/SFX --name my_library

vocalocate-pipeline --lake lake report            # data-quality summary
vocalocate-pipeline --lake lake report --sql "SELECT split, count(*) FROM v_training_pairs GROUP BY split"
```

Settings live in [`configs/default.toml`](configs/default.toml); pass another
file with `--config` for a different environment.

## Tests

```bash
cd pipeline && python -m pytest -q
```

The tests build small synthetic datasets laid out exactly like ESC-50,
VimSketch, the QVIM DEV set and team recordings, so they run offline in a few
seconds.

## On AWS

The lake's source of truth on AWS is an S3 bucket with the same layout. A job
pulls what it needs into a local working copy, runs the normal flow and pushes
back only what it produced (`remote.py`, `sync.py`). An ingest uploads only its
own source's files; `load-warehouse` then rebuilds the warehouse from the whole
lake, so runs of different sources can overlap safely:

```bash
# What the Fargate task runs; the task definition sets these three variables.
export VOCALOCATE_CONFIG=configs/prod.toml VOCALOCATE_RAW_URI=s3://<raw bucket> VOCALOCATE_LAKE_URI=s3://<lake bucket>
vocalocate-pipeline remote ingest --source esc50      # raw files from s3://<raw bucket>/esc50/
vocalocate-pipeline remote load-warehouse
vocalocate-pipeline remote report

vocalocate-pipeline sync s3://<lake bucket> lake --only warehouse   # copy part of the lake by hand
```

`configs/dev.toml` and `configs/prod.toml` are the AWS environments' settings.
They change only the lake location and the worker count, never the audio,
feature or split settings. The container image is built from `Dockerfile`.
Deployment, environments and triggers are described in
[`../infra/README.md`](../infra/README.md).

## Not implemented yet

See the "Next steps" section of the A2 write-up. In short: speech filtering for library
files, Freesound and freesound-laion-640k ingestion, augmentation, FAISS indexing, and
a parity test against the torchaudio feature code.

Hard-negative mining and encoder embeddings now have tables here
(`hard_negatives`, `embeddings`, plus `sfx_metadata` and the `sfx_meta`
ingester); the code that fills them is in [`../demo`](../demo).
