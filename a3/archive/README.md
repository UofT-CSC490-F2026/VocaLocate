# A3 archive

Everything tried for A3 that is not part of the submission: xIPReLU, Depth-Attention, MGVR, XSA, MUDD-lite, SSA, mHC,
plain looped-l2, looped-u7, looped-u9, looped-rand13 and Diff V2 + XSA, with their code, tests, docs, logs and results.

This branch merges every A3 branch as it stood before the cleanup, so all of that code still runs here.

- `a3/nanochat/`: code with all the ablations and their tests
- `a3/modal/`: run scripts, logs, results, figures and offline W&B runs
- `a3/modal/part_2_final_result_files/`: the 4×H100 Part 2 runs
- `a3/archive/mhc/`: mHC was never committed to a branch, so its patch, test and logs are saved here
- `a3/archive/local-logs/`: local copies of logs that differ from the committed ones, often only in line endings

W&B project: https://forge.coreweave.com/wandb/tanayjyot-university-of-toronto/a3-archive
