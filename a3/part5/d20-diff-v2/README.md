# Part 5: d20 Diff V2 run

One d20 run with `--experiment=diff-v2`, seed 42, on 8× H100 80GB (`--nproc_per_node=8 --device-batch-size=8`).
It trained for 9,960 steps × 524,288 tokens = 5.22B tokens, using code `be8e30c`, the team tokenizer and 130 ClimbMix shards, so no data was repeated.

| Metric | Value |
|---|---|
| Final val bpb (also the minimum) | **0.7325** |
| CORE | **0.2573** |
| FLOPs per token | 3.36e9 (1.16× the d20 baseline's 2.89e9) |
| Throughput | 955k tok/s, 40.6% MFU |
| Peak memory | 29.7 GB per GPU |
| Training time | 97.3 min |

Full command:

```sh
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --experiment=diff-v2 --depth=20 \
  --num-iterations=9960 --total-batch-size=524288 --device-batch-size=8 --max-seq-len=2048 \
  --eval-every=500 --core-metric-every=999999 --sample-every=-1 --save-every=500 \
  --run=d20-diff-v2 --model-tag=d20-diff-v2
```

## Three attempts

The run was stopped twice, at steps 2,845 and 6,496. Both times my local Modal client's login token expired, and that stopped the app. It resumed on the same 8 GPUs from the step-2,500 and step-6,500 checkpoints. Each resumed step reproduced the checkpoint's val bpb exactly (step 2,500: 0.840183 both times), so the result is unaffected. About 450 steps were trained twice. The figures colour each attempt separately and mark the restarts with dotted lines.

## Figures

| File | Shows |
|---|---|
| `figures/d20_val_bpb.png` | Validation bpb every 500 steps. Step 0 (3.169) is left out so the curve is readable. |
| `figures/d20_train_loss.png` | Training loss per step (nanochat's smoothed loss, from step 50 on) |
| `figures/d20_throughput.png` | Tokens per second (rolling median of 25 steps) |
| `figures/d20_lr_schedule.png` | Learning-rate multiplier: warmup, constant, then linear decay |
| `figures/d20_core_tasks.png` | CORE per task (centered accuracy, 500 items per task), with the overall CORE marked |
| `figures/d20_curves_summary.json` | The val bpb at each eval step, merged across attempts |

These figures use the same metrics the run logged to W&B. They are built from the training log so that anyone can rebuild them without W&B access:

```sh
python plot_d20.py logs/d20-diff-v2.log results/d20-diff-v2.json figures
```

The plotting script needs matplotlib.

## Files

- `logs/d20-diff-v2.log`: the full stdout of all three attempts. Each attempt starts with a `# attempt at` line.
- `results/d20-diff-v2.json`: the final metrics, the val curve, per-task CORE and the command.

## W&B

The W&B runs are at <https://forge.coreweave.com/wandb/riyadvaliyev/nanochat-d20-ablations>, in group `d20-ablations`. There is one run per attempt:

| Run | Steps |
|---|---|
| `njc7kbua` | 0–2,845 (upload ends slightly early, because the offline file was cut off when the app stopped) |
| `53r4h6w4` | 2,500–6,496 |
| `q8pjra7z` | 6,500–9,960, plus CORE |

## Scaling check

d12 Diff V2 (3-seed mean) reached 0.8412 val bpb, and d20 reached 0.7325. With the plan's formula, bpb_d20 = bpb_d12 × (N20/N12)^−α, where N is the baseline scaling parameter count (110.1M → 435.2M), that gives α ≈ 0.101. This sits between the plan's 0.076 and Compos3D's 0.1188.
