# Part 4 context extension results

The baseline Picochat trains at a sequence length of 512 tokens to step 1260
(checkpoint A), then resumes at 2048 tokens to step 2520 (checkpoint B).
Both stages use one continuous learning-rate schedule, seed 42, and 524,288
tokens per optimizer step. Training uses ClimbMix training shards 0-31 and the
separate validation shard 6542. The tokenizer is the unchanged copy in
`../tokenizer`, inherited from commit `be8e30c`.

## Training

From the repository root, with Modal configured:

```sh
modal run a3/modal/train_part4.py::prep
modal run a3/modal/train_part4.py::smoke
modal run --detach a3/modal/train_part4.py::train
```

The launcher verifies the tokenizer hashes and dataset manifest before training.
Checkpoints are stored on the `vocalocate-a3-part4` Modal volume under
`/part4/base_checkpoints/d12-context-part4/`. Model and optimizer weights are
excluded from Git. The `checkpoints/` directory contains the saved configuration,
validation metric, and data-loader position for A and B.

## Local evaluation

Use the dependencies declared in `a3/nanochat/pyproject.toml`. Download
`model_001260.pt`, `meta_001260.json`, `model_002520.pt`, and `meta_002520.json`
from the volume into `a3/nanochat/dev-ignore/part4/`. From `a3/nanochat`, run:

```sh
python -m scripts.part4_eval --checkpoint-dir dev-ignore/part4 --steps 1260 2520 --trials 128 --seed 42 --output ../part4/results/passkey-results.json
python -m scripts.part4_eval --checkpoint-dir dev-ignore/part4 --steps 1260 2520 --trials 128 --seed 20261104 --competing-records --output ../part4/results/passkey-competing-paired.json
```

Both checkpoints receive identical prompts and candidate words within each task.
The code compares the next-token scores of four candidate words. A correct
answer scores one; an incorrect answer scores zero. Each condition has 128
prompts. Input length and the distance from the target code to the final input
token are controlled separately. Both checkpoints are evaluated with an input
limit of 2048 tokens.

To repeat the two input-edit checks using checkpoint B, from `a3/nanochat`:

```sh
python -m scripts.part4_eval --checkpoint-dir dev-ignore/part4 --steps 2520 --trials 128 --seed 42 --prompt-checks --output ../part4/results/passkey-leakage-audit.json
```

This repeats the original single-record prompts, then scores the versions with
the code replaced by `unknown` or the next candidate. The candidate list stays
fixed. Removing the code keeps the original expected answer; substituting a
candidate makes that replacement the expected answer. No model is trained.

## Part 2 baseline reference

Place Tanay's `d12-baseline` files `model_002520.pt` and `meta_002520.json` in
`a3/nanochat/dev-ignore/part2-baseline/`. This directory is excluded from Git;
the supplied checkpoint files are not included in the repository. From
`a3/nanochat`, evaluate the baseline on the same prompts as A and B:

```sh
python -m scripts.part4_eval --checkpoint-dir dev-ignore/part2-baseline --steps 2520 --trials 128 --batch-size 2 --seed 42 --output ../part4/results/part2-baseline-single-record.json
python -m scripts.part4_eval --checkpoint-dir dev-ignore/part2-baseline --steps 2520 --trials 128 --batch-size 2 --seed 20261104 --competing-records --output ../part4/results/part2-baseline-four-record-paired.json
```

This baseline trained at 2048 tokens throughout, with the same 2520 optimizer
steps and 1,321,205,760 training tokens as B. The single-record output contains
accuracy totals; the four-record output also contains each prediction.
`results/part2-baseline-comparison.json` and `.csv` collect A, B, and baseline
accuracy in all nine conditions, together with B-minus-baseline paired
intervals. The JSON records the interval formula and source filenames.
`results/part2-baseline-eval-manifest.json` records the checkpoint and tokenizer
hashes, evaluation commands, runtime, and output hashes. A runtime cross-check
reproduced all 256 archived B predictions at input length 2048 and code
distances 128 and 1536. These are inference results; no model was trained.

## Rebuild statistics and figures

From `a3/nanochat`, the saved per-trial predictions and training logs are enough
to recalculate all nine A/B paired confidence intervals and regenerate the
three A/B report figures. Checkpoint weights and a GPU are not needed:

```sh
python -m scripts.part4_analyze
```

The script first checks that paired trial seeds, candidate lists, and expected
answers match, and that per-trial predictions reproduce each saved total. It
writes `results/passkey-paired-summary.json` and three files in `figures/`:
`training-validation-loss.png`, `four-record-accuracy.png`, and
`paired-accuracy-difference.png`. The last figure shows all nine B-minus-A gains,
their approximate 95% paired intervals, and a zero-difference reference line.
Matplotlib, listed in the nanochat development dependencies, is required for
plots. Use `--summary-only` to recalculate just the statistics without it.

To also rebuild the baseline comparison JSON and CSV and Figure 4.4
(`figures/baseline-accuracy-difference.png`), use:

```sh
python -m scripts.part4_analyze --include-baseline
```

The added figure shows B minus the Part 2 baseline in all nine four-record
conditions, with paired intervals calculated from their matched predictions.
The baseline flag can also be combined with `--summary-only`.

To write a separate copy without replacing the archived outputs:

```sh
python -m scripts.part4_analyze --include-baseline --output-dir dev-ignore/part4-rebuilt
```

## Saved outputs

- `results/passkey-results.json`: single-record accuracy for A and B.
- `results/passkey-competing-records.json`: original four-record accuracy summary.
- `results/passkey-competing-paired.json`: matching four-record totals with each
  trial's seed, candidate token IDs, expected answer, and prediction.
- `results/passkey-paired-summary.json`: paired differences in four-record
  accuracy and approximate 95% confidence intervals, in percentage points.
- `results/passkey-leakage-audit.json`: checks on B after replacing the stated
  code with `unknown` or a different candidate. `no_answer_correct` counts
  selection of the removed original code, not selection of `unknown`.
- `logs/`: the original training logs for the two stages.
- `figures/`: the training/validation curves, four-record accuracy, and paired
  A/B and B/baseline accuracy differences used in the report.

For each condition, the confidence interval uses paired correctness differences
`d_i = correct_B_i - correct_A_i`, with values -1, 0, or 1. The estimated gain is
`100 * mean(d)`, and the interval is `100 * (mean(d) +/- 1.96 * stdev(d) / sqrt(n))`,
using the sample standard deviation and `n = 128`.
Each interval applies to one condition and describes variation across prompts
for these two fixed checkpoints.

## Tests

From `a3/nanochat`:

```sh
python -m pytest tests/test_part4_eval.py tests/test_part4_analyze.py tests/ablations/test_context_resume.py -q
```
