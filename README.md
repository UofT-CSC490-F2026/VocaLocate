# VocaLocate

VocaLocate lets a video editor or sound designer find a sound effect in their own library by
imitating it with their voice ("pshhh", "bonk"), optionally adding a short text hint such as
"metallic". This repository holds the CSC490 project: the data pipeline that prepares audio for
the search model, the AWS infrastructure that runs it, and the model experiments and local search
app built on top.

## Where each assignment part lives

| A2 part | What it is | Where |
| --- | --- | --- |
| 1. Aspirational datasets | Wish-list schemas | The report; the real tables are in [`pipeline/vocalocate_pipeline/schemas.py`](pipeline/vocalocate_pipeline/schemas.py), and one warehouse view per Part One schema (`v_sfx_file`, `v_retrieval_query`, ...) in [`warehouse.py`](pipeline/vocalocate_pipeline/warehouse.py) |
| 2. Reality check | Datasets we can use | The report; one ingester per chosen dataset in [`pipeline/vocalocate_pipeline/ingest/`](pipeline/vocalocate_pipeline/ingest) |
| 3. Data-processing pipeline | Ingest, clean, transform, warehouse | [`pipeline/`](pipeline) |
| 4. Infrastructure as code | Terraform for AWS, dev and prod | [`infra/`](infra) |
| 5. Disaster recovery | Delete prod and restore it | [`infra/disaster-recovery.md`](infra/disaster-recovery.md) |
| (A1) | Project proposal | [`a1/a1.pdf`](a1/a1.pdf) |

## Repository map

```
VocaLocate/
├── pipeline/                     Part 3: the data pipeline (Python package vocalocate_pipeline)
│   ├── vocalocate_pipeline/      the code, read in this order:
│   │   ├── cli.py                  the vocalocate-pipeline command and its sub-commands
│   │   ├── config.py               every tunable setting, and the config fingerprint
│   │   ├── ingest/                 1. Ingest: one small reader per dataset (see below)
│   │   ├── flow_functions/         the per-clip steps the flows run:
│   │   │   ├── audio.py              2. Clean: mono, 32 kHz, trim, loudness; reject or flag bad clips
│   │   │   ├── features.py           3. Transform: log-mel features, as the QVIM baseline computes them
│   │   │   └── splits.py                train/val/test by a seeded hash of each reference sound
│   │   ├── flows.py                the two flows (dataset, library) that run stages 1-4 with caching
│   │   ├── schemas.py              the columns and types of every lake table
│   │   ├── lake.py                 4. Load: where each file goes in bronze/silver/gold
│   │   ├── warehouse.py               the DuckDB warehouse and its SQL views
│   │   ├── sync.py                 copy lake files between a local folder and S3
│   │   └── remote.py               what the AWS tasks run: pull from S3, run a flow, push back
│   ├── tests/                    pytest suite on small fake datasets (no downloads)
│   ├── configs/                  settings: default.toml, dev.toml, prod.toml
│   ├── docs/diagrams/            architecture, AWS deployment and library-flow figures
│   ├── Dockerfile                the container image the AWS tasks run
│   └── README.md                 pipeline details: layers, tables, views, commands
├── infra/                        Parts 4-5: Terraform
│   ├── bootstrap/                once per account: state bucket, OIDC providers, budget
│   ├── backup/                   prod's backup buckets in us-east-2 (separate state)
│   ├── envs/dev/, envs/prod/     one short root per environment
│   ├── modules/                  vocalocate_stack (one environment) and the modules it wires:
│   │                             bucket, network, data_lake, pipeline_compute, orchestration,
│   │                             warehouse_catalog, monitoring, access
│   ├── disaster-recovery.md      the Part 5 runbook
│   └── README.md                 deploying and running the pipeline on AWS
├── demo/                         model training, evaluation and the local search app
│   ├── src/vocalocate_demo/      encoders, negative mining, CLAP text branch, fusion, app
│   ├── modal/recipe_app.py       parallel GPU training on Modal, reading the S3 lake
│   └── README.md                 experiments and results
├── a1/a1.pdf                     the A1 proposal
└── .github/workflows/            CI: pipeline tests, demo tests, Terraform checks, image build
```

### The ingesters

Every dataset has its own way of saying which file is which, so each gets a small module in
[`pipeline/vocalocate_pipeline/ingest/`](pipeline/vocalocate_pipeline/ingest) that only parses that
layout. The shared work (reading index files, recording files, missing files, query and caption
IDs) lives in `IngestBuilder` in [`ingest/__init__.py`](pipeline/vocalocate_pipeline/ingest/__init__.py).

| Module | Dataset | Used for |
| --- | --- | --- |
| `vocal_imitation_set.py` (`vis`) | Vocal Imitation Set v1.1 | Main training set: imitation and reference pairs, distractors, text hints |
| `vimsketch.py` | VimSketch (+ VocalSketch excluded takes) | Broader training set; what the QVIM baseline trained on |
| `qvim_dev.py`, `qvim_eval.py` | QVIM challenge DEV and EVAL | Held-out benchmark (always the test split) |
| `captdure.py` | CAPTDURE | Captions to test the CLAP text branch (always the test split) |
| `esc50.py` | ESC-50 | Fast smoke test of the whole pipeline |
| `team.py` | Our own recordings (`recordings.csv`) | Voice + text + target queries for tuning the fusion weight |
| `sfx_meta.py` | Any set with a `metadata.csv` | The generic path; VIS and VimSketch convert to its rows |

## Find things fast

| I want to... | Look at |
| --- | --- |
| Change a cleaning threshold or feature setting | `pipeline/configs/*.toml` and `pipeline/vocalocate_pipeline/config.py` |
| Change how a clip is cleaned, featurised or split | `pipeline/vocalocate_pipeline/flow_functions/` |
| Add a dataset | A new module in `pipeline/vocalocate_pipeline/ingest/`, registered in `ingest/__init__.py` |
| See or change a table's columns | `pipeline/vocalocate_pipeline/schemas.py` |
| Add a SQL view | `VIEWS` in `pipeline/vocalocate_pipeline/warehouse.py` |
| Understand what runs on AWS and when | `infra/modules/orchestration/main.tf` (the Step Functions state machine) |
| Change dev or prod sizes | `infra/envs/dev/main.tf`, `infra/envs/prod/main.tf` |
| Change permissions | `infra/modules/access/` (CI and Modal), `infra/modules/pipeline_compute/` (the task) |

## Quick start

```bash
cd pipeline
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q                       # the whole suite on fake datasets, no downloads

# Real audio: ESC-50 (about 600 MB)
git clone https://github.com/karolpiczak/ESC-50
vocalocate-pipeline --lake lake --workers 4 -v ingest-dataset --source esc50 --path ESC-50
vocalocate-pipeline --lake lake report    # data-quality summary from the warehouse

# Your own sound folder, indexed incrementally
vocalocate-pipeline --lake lake index-library --root ~/SFX --name my_library
```

Deploying to AWS is covered in [`infra/README.md`](infra/README.md); the search app and
experiments in [`demo/README.md`](demo/README.md).

## How data flows

```
datasets / team recordings / editor folders
   │  1. ingest     ingest/*.py                  → bronze: raw_files manifest (raw audio never modified)
   │  2. clean      flow_functions/audio.py      → silver: clips table + clean 32 kHz mono WAV
   │  3. transform  flow_functions/features.py,  → gold:   log-mel features, splits
   │                flow_functions/splits.py        (pairs and queries in silver)
   ▼  4. load       warehouse.py                 → warehouse/vocalocate.duckdb with SQL views
training (v_training_pairs) · fusion tuning (v_retrieval_query) · search app (v_library_index)
· data-quality report (v_clip_quality)
```

On AWS the same code runs in ECS Fargate tasks started by Step Functions, against an S3 lake with
the same layout; see the diagram in [`infra/README.md`](infra/README.md).
