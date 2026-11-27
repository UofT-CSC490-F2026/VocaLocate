# Re-indexing cache debugging demo

This standalone script is for the A5 walkthrough and is not included in the GitHub CI tests. Justin's automated regression test remains in `demo/tests/test_app.py`.

It copies the five VimSketch SFX into temporary folders, runs the real library-indexing pipeline, then searches after adding, removing, or replacing a file. It checks that scores are recomputed for the current index and reused on an unchanged index. Original sample files are not changed.

Voice embeddings and CLAP scores are stand-ins. Fake CLAP is deterministic but its scores do not measure semantic similarity; the vocal imitation and UI are not exercised. No model checkpoints are needed. Requires the repository Python dependencies installed.

From the repository root:

```bash
.venv/bin/python a5/a5-part-5-clap-cache-bug/clap-cache-test-with-sample-folder/debug_reindex.py
.venv/bin/python a5/a5-part-5-clap-cache-bug/clap-cache-test-with-sample-folder/debug_reindex.py --scenario remove
```

Run one scenario at a time when demonstrating the old implementation, because an expected failure stops the script. On the fixed branch all three should pass. The script prints filenames and simulated text scores; inspect the error or assertion on the old branch rather than treating the fake scores as retrieval quality.

To compare the current local `main` commit with the current `HEAD` commit without changing branches:

```bash
.venv/bin/python a5/a5-part-5-clap-cache-bug/clap-cache-test-with-sample-folder/cache_bug_before_and_after.py
```

Add `--scenario remove` (or `add` or `replace`) to run one scenario. The launcher prints both commit IDs, extracts only their tracked Python packages to temporary directories, and removes them afterwards. It does not fetch remote updates or include uncommitted code changes. If `main` later contains the fix, both versions may pass. Failures on main should be inspected: a failed run can have causes other than the cache bug. The command exits nonzero if an after scenario fails or setup fails.

Each scenario lists the original samples folder, temporary library folder, starting sounds, and changed files. A NumPy shape error reports the current voice scores first and the stale text scores second, following the actual calculation; its order is not chronological.

The add case adds the church bell; the remove case removes the dog bark; the replacement case removes the dog bark and adds the church bell. Each case starts from a fresh temporary library.
