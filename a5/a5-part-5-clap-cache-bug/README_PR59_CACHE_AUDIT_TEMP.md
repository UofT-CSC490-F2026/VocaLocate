# Temporary audit of PR 59 CLAP text-score caching

This is a working note for A5 Part 5, based on an isolated inspection and test of [PR #59](https://github.com/UofT-CSC490-F2026/VocaLocate/pull/59), head `7b713596ad5e3e5457fc45b4ee0954067fc2c917`. Results apply to that version and should be rerun if the PR changes.

## Summary

PR #59 clears cached text scores when it publishes a newly indexed library and recomputes scores if a cached score vector has the wrong length. This addresses the sequential add, remove, replace, and reorder cases tested below. A controlled overlapping-search test nevertheless reproduced stale score reuse: an old search can repopulate the shared cache after the new index is published and its cache is cleared.

## Sequential scenarios

| Scenario | Result on PR #59 | Notes |
| --- | --- | --- |
| Add a sound | Passed | New index has more files. |
| Remove a sound | Passed | New index has fewer files. |
| Replace a sound, keeping the same count | Passed | Fresh scores matched the current indexed files. |
| Simulated index reorder | Passed after harness adaptation | The temporary test imported `LibraryIndex` from `search` rather than `app`, reflecting PR #59's changed module structure. This was a simulated index permutation; rearranging Finder icons does not change the app's path-sorted index. |

The scenarios used sample WAVs and PR #59's indexing and cache code with simulated voice embeddings and fake CLAP outputs. The UI and real CLAP retrieval quality were not tested. Five relevant tests from PR #59 also passed in isolation. Its existing tests cover reindexing and missing clips, but no direct equivalent of the complete four-scenario cached-hint sequence was identified.

## Overlapping search and reindex

A deterministic isolated test used PR #59's actual `AppState.search` and `_index` control flow, while mocking indexing, voice embeddings, and text similarity scores. It forced this sequence:

1. An old-index search begins and computes a simulated score of `0.25`.
2. A same-size replacement index is published and the shared cache is cleared.
3. The old search resumes and writes its old scores to the shared cache.
4. A new-index search reuses `0.25` for `new.wav`, although its freshly computed simulated score would be `0.75`.

The observed output was `new.wav text score 0.25 expected 0.75 reused ['query embedding', 'text scores']`. This is a reproducible backend timing issue in the controlled setup, not a full browser or real-model test. The relevant logic is in [PR #59 app.py, index publication and search](https://github.com/UofT-CSC490-F2026/VocaLocate/blob/7b713596ad5e3e5457fc45b4ee0954067fc2c917/demo/src/vocalocate_demo/app.py#L550-L608).

## Implications and next steps

- Keep the existing add, remove, same-size replace and reorder regression scenarios; adapt their harness to PR #59's API where needed.
- Add a deterministic regression test in which a search overlaps reindexing and finishes scoring the old index after the new index is published.
- Consider an index-scoped cache, as in [Justin's PR #50](https://github.com/UofT-CSC490-F2026/VocaLocate/pull/50), or another approach that associates cache reads/writes with the correct index generation. Validate whichever implementation is integrated; do not assume a transplant alone resolves all concurrency cases.
- Re-run the sequential and overlap tests against the final combined branch. Separately verify browser behavior and real CLAP if needed.

The existing sample-library script is `clap-cache-test-with-sample-folder/debug_reindex.py`. It uses real sample files but simulated model outputs, and is not a UI test. The isolated race proof used during this audit was written outside the repository and is temporary; its sequence is recorded above rather than presented as a committed test.
