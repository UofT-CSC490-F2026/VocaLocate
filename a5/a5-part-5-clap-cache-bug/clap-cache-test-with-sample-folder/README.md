# CLAP cache bug demo

This standalone A5 demo reproduces stale text scores after a library changes. It is not run in CI; the automated regression test is in `demo/tests/test_app.py`.

## Run

From the repository root, with the project dependencies installed:

```bash
.venv/bin/python a5/a5-part-5-clap-cache-bug/clap-cache-test-with-sample-folder/cache_bug_before_and_after.py
```

Add `--scenario add`, `remove`, `replace`, or `reorder` to run one case. The launcher compares committed local `main` with committed `HEAD` without switching branches. It prints the commit IDs, uses temporary source copies, and does not fetch updates or include uncommitted changes. If `main` gains the fix, both versions may pass. It exits nonzero if setup or an after case fails; inspect before failures rather than assuming their cause.

To run only the current checkout, use `debug_reindex.py` in the same folder with the same optional `--scenario` argument.

## Scenarios

Each case starts independently with thunder, church bell, waterfall and door slam, and the text hint “metallic ringing.”

- **Add:** add dog bark (4 → 5).
- **Remove:** remove church bell (4 → 3).
- **Replace:** replace church bell with dog bark (4 → 4).
- **Reorder:** simulate a different order for the same four indexed sounds. The app sorts by path; rearranging Finder icons does not change the index.

## What the demo measures

The script copies real VimSketch SFX into temporary libraries and runs the real indexing and cache logic. It leaves the original files untouched. The vocal imitation, voice embeddings and CLAP outputs are simulated; the UI is not tested, and the fake scores do not measure sound similarity. No model checkpoints are needed.

When a search returns results, the demo compares the fake CLAP audio embeddings behind its scores and the scores themselves against a fresh calculation for the current files.

For a size mismatch, NumPy reports the current voice-score shape first and the stale text-score shape second; that order is not chronological.
