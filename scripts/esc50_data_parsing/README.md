# ESC-50-Voice preparation

Creates one CSV row per vocal imitation, paired with its ESC-50 target. Uses
Python 3.10+ and the standard library; extract both datasets first.

Run from the repository root, replacing the input paths:

```bash
python3 scripts/esc50_data_parsing/parse_esc50_voice.py \
  --path /path/to/vocal_imitation \
  --esc50-path /path/to/ESC-50
```

Output defaults to this folder's `output/raw_paired_manifest.csv`. Override it
with `--out /path/to/file.csv`; relative overrides use your working directory.

```text
voice_file, sfx_file, imitator_id, category, recording_id
```

- IDs use `esc50_voice:000` and `freesound:100032`.
- Rows sort by numeric Freesound ID, target filename, then speaker ID.
- `voice_file` is relative to the voice folder: `000/<voice>.wav`.
  `sfx_file` is relative to the ESC-50 root: `audio/<target>.wav`.
- The parser does not generate `output/vis_overlap_report.csv`; that is a saved
  audit. Integration steps are in [ESC50_VIMSKETCH_PLAN.md](ESC50_VIMSKETCH_PLAN.md).

Tests use synthetic files and require no dataset download:

```bash
python3 -m unittest discover -s scripts/esc50_data_parsing/tests -v
```
