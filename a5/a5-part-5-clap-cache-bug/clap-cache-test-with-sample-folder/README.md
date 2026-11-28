# CLAP cache bug demo

This standalone A5 demo reproduces stale text scores after a library changes. It is not run in CI; the automated regression test is in `demo/tests/test_app.py`.

## Run

From the repository root, with the project dependencies installed:

```bash
.venv/bin/python a5/a5-part-5-clap-cache-bug/clap-cache-test-with-sample-folder/cache_bug_before_and_after.py
```

Add `--scenario add`, `remove`, `replace`, or `reorder` to run one case. The launcher compares the pinned pre-fix commit `93f103ea6ff1a327995ea7096f74c54d5821749d` with committed `HEAD` without switching branches. It prints the commit IDs, uses temporary source copies, and does not fetch updates or include uncommitted changes. It exits nonzero if setup or an after case fails; inspect before failures rather than assuming their cause.

To run only the current checkout, use `debug_reindex.py` in the same folder with the same optional `--scenario` argument.

To run the historical pre-fix **web UI** from the same commit (not the newer desktop UI), supply a compatible voice-model folder and a separate data-lake directory:

```bash
.venv/bin/python a5/a5-part-5-clap-cache-bug/clap-cache-test-with-sample-folder/launch_before_ui.py --models /absolute/path/to/models --lake /absolute/path/to/test-lake
```

Open `http://127.0.0.1:8765`, and press Ctrl+C in the terminal when finished. The default uses real CLAP; add `--clap-backend fake` for simulated scores. A trained voice checkpoint is required either way. The pre-fix code is deliberately buggy; use copied sample files and an isolated test lake.

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
