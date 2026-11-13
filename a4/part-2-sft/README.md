# A4 Nanochat checkpoints and Modal files

This README explains the A4 Part 2 Modal launcher, the files downloaded from the Modal volumes, and which artifacts later assignment parts are likely to need. Check each part's handout before launching another run; the guidance below describes the current artifact layout and intended checkpoint lineage.

## Directory layout

```text
a4/
├── modal/
│   ├── original-d20-model-files/
│   │   ├── model_009960.pt
│   │   └── meta_009960.json
│   └── original-d20-sft-modal-files/
│       ├── logs/a4-original-sft.log
│       ├── results/a4-original-sft.json
│       └── sft_runs/a4-original-sft/chatsft_checkpoints/d20-baseline-8-H100/
│           ├── model_000932.pt
│           └── meta_000932.json
└── part-2-sft/
    ├── modal_chat_sft.py
    ├── modal_a4.py
    ├── MODAL_SETUP_LOG.md
    └── README.md
```

The source Modal volume paths are under `/a4` in `a4-nanochat-volume`. These local folders contain copies for versioning and teammate access. The baseline optimizer shards are in the Modal volume but were not downloaded here. The SFT optimizer shards were intentionally omitted from the local copy.

## Launcher: `modal_chat_sft.py`

`modal_chat_sft.py` is a local Modal entrypoint. It mounts the named volume at `/vol`, checks for the requested input checkpoint and tokenizer, creates a run-specific directory, and invokes nanochat's `scripts/chat_sft.py` in a Modal container. It selects the requested GPU count, can load optimizer state, passes W&B settings, streams training output to a log, and writes a run summary JSON.

The launcher does not edit the tracked `a3/nanochat/scripts/chat_sft.py`. It adjusts a container-only copy for W&B project selection and processed-token logging. The launcher currently runs the SFT script. Do not assume it also ran a separate midtraining stage; check the training command, logs, and output checkpoints for each required stage.

`modal_a4.py` prepares the A4 volume and copies the pinned A3 baseline checkpoint into it. `MODAL_SETUP_LOG.md` records setup decisions, file hashes, and audit notes. Use the A4 volume for Modal execution; local copies are for sharing and artifact review.

## Downloaded baseline files

`a4/modal/original-d20-model-files/` contains the A3 pretrained d20 baseline at step 9960:

- `model_009960.pt` contains pretrained model weights.
- `meta_009960.json` contains the model configuration and checkpoint metadata needed to reconstruct the model.

They must be kept as a matching pair. The model SHA-256 is `0bb5c834e1a111e963ca3678a26fd57d6185ae97f836baf8d69bc86a00c9dea1`; the metadata SHA-256 is `4b40c46d5e2af19d095bc13f20b73e6e3cd2eb5e07b1c8e335509abe30e1620e`.

The corresponding pretrained optimizer shards `optim_009960_rank0.pt` through `optim_009960_rank7.pt` remain on the source/A4 Modal volumes. They are needed only when a training run restores the optimizer state from this exact baseline checkpoint with a compatible eight-rank configuration. They are not required to load the weights for inference or evaluation.

## Downloaded original-data SFT files

`a4/modal/original-d20-sft-modal-files/` contains the completed A4 original-data SFT run:

- `logs/a4-original-sft.log` contains the command, progress output, validation and task evaluations, checkpoint save messages, and W&B sync summary.
- `results/a4-original-sft.json` records operational provenance: input and output checkpoint paths, step, GPU count, optimizer-loading flag, W&B project/entity, return code, duration, and log path. It is not the metrics history export. The custom launcher writes this JSON; nanochat's SFT script emits training/evaluation metrics to stdout and W&B.
- `sft_runs/.../model_000932.pt` contains the fine-tuned weights.
- `sft_runs/.../meta_000932.json` is the matching SFT checkpoint metadata and configuration. Keep it with the model file.

The run completed successfully with return code 0 and produced checkpoint step 932. The log and result JSON link to W&B run `aa54srb7` in project `a4-nanochat-rl`.

The SFT optimizer shards `optim_000932_rank0.pt` through `optim_000932_rank7.pt` exist on the A4 Modal volume, but were omitted locally because ordinary model loading/evaluation uses the model and metadata, and the planned expanded-data experiment starts from the original A3 baseline. Retain or retrieve these shards if you plan to resume training from this SFT checkpoint and restore its optimizer state.

## What later assignment work should use

The assignment handout determines the exact experiment. Preserve checkpoint lineage and record which checkpoint, tokenizer, code revision, dataset, optimizer state, and hardware each run uses.

| Assignment work | Starting artifacts | Notes |
|---|---|---|
| Part 2(a), pretrained comparison | A3 `model_009960.pt` + `meta_009960.json`; shared A3 tokenizer | Evaluate the pretrained baseline and post-training checkpoints using the same evaluation code and metrics. For training, load the A3 optimizer shards if using the intended eight-rank warm start. |
| Part 2(a), midtraining and SFT stages | Pretrained A3 model and metadata; tokenizer; matching A3 optimizer shards when restoring optimizer state | Keep midtraining and SFT outputs identifiable as separate stages. This launcher invokes `chat_sft.py`; verify the separate midtraining command/checkpoint and do not report SFT alone as both stages. |
| Part 2(b), expanded dataset | Start from the same A3 pretrained model, metadata, tokenizer, and (if chosen) A3 optimizer shards as the corresponding original-data run | This isolates the dataset change. Do not initialize the expanded run from the original SFT output unless the experiment is explicitly designed as sequential fine-tuning. Use a distinct run tag and document data/config changes. |
| Part 3, GSM8K/RL | Typically the completed Part 2 post-training checkpoint required by the handout, plus its metadata and tokenizer | Confirm whether the part specifies the original-data or expanded-data branch. Use the SFT-produced optimizer shards only if resuming training from that SFT checkpoint with optimizer restoration. |
| Parts 4 and 5 | The checkpoint lineage named by each part's handout; include its matching metadata and tokenizer | Evaluation/inference generally needs model, metadata, tokenizer, and evaluation code/data. If a part continues training, determine its prescribed optimizer initialization and retrieve matching shards if required. Do not assume every later part starts from the same SFT model. |

For model evaluation or inference, optimizer shards can usually be left out. For optimizer-resumed training, the shards must correspond to the exact model checkpoint and distributed world size; do not substitute shards from another checkpoint. If a run starts from A3 baseline, use its A3 optimizer shards (or consistently choose a fresh optimizer as an explicit experimental choice). If a run starts from SFT step 932 and restores optimizer state, use the SFT step-932 shards.

## Sharing and Git

The model `.pt` files are multi-gigabyte artifacts. Prefer Git LFS or a model artifact store over regular Git blobs, and record download locations and checksums for collaborators. Optimizer shards are also large and can be shared separately when a later training step needs them. Keep credentials and secret values out of the repository. Review logs and JSON before publishing for sensitive values or machine-specific paths.
