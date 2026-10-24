# MOSS acoustic tag generation

`moss.py` analyzes a folder of audio **recursively**, using
`OpenMOSS-Team/MOSS-Audio-4B-Instruct`. It considers the likely sound source/event
as context, then generates acoustic tags describing pitch, timbre, texture,
reverberation, and temporal qualities. Tags should describe what is audible,
rather than repeat source names.

The workflow is:

1. Load the complete task and output instructions from a prompt file.
2. The prompt asks MOSS to **generate acoustic tags directly from each recording**.
   Python then parses the tags MOSS generated into annotation fields.
3. If the response contains text but no parsed tags, use a text-only fallback to
   extract acoustic descriptors already present in that response, using
   `prompts/tag_extraction.txt`. The main pass
   already requested tags; this fallback salvages missing tag fields without
   listening to the audio again or checking the description's accuracy.
4. When `--html-report PATH` is supplied, call `html_report.py` to produce a
   human-review table: **File, Status, Acoustic tags, Raw output**. Raw responses
   are displayed directly, including any text-only recovery response.

Annotations are candidates for review. Status `success` indicates accepted
formatting, not verified acoustic accuracy; `fail` indicates an invalid response
or a processing error. Error details remain visible in the raw-output column.

## Local use

Activate an environment with MOSS and its runtime dependencies installed:

```bash
conda activate moss-audio-env
```

For a fresh environment, follow the [MOSS installation instructions](https://github.com/OpenMOSS/MOSS-Audio#environment-setup).
The first run downloads model weights from Hugging Face; `--model /path/to/weights`
uses an existing local copy. Optional authentication is read from `HF_TOKEN`.

From this directory:

```bash
# Analyze audio and print progress; no report file is created by default.
python moss.py /path/to/sfx

# Generate an HTML report for human review.
python moss.py /path/to/sfx \
  --prompt-file prompts/audio_tag_extraction_instruction.txt \
  --recovery-prompt-file prompts/tag_extraction.txt \
  --html-report reports/moss-review.html \
  --overwrite
```

The script reads the entire UTF-8 prompt file, including task and output sections,
without appending extra instructions. If omitted, `--prompt-file` defaults to
`prompts/audio_tag_extraction_instruction.txt` beside the script.

`--recovery-prompt-file` selects the separate text-only fallback instructions;
it defaults to `prompts/tag_extraction.txt` beside the script. It follows the
audio prompt's tagging rules (4–5 when supported, ranked by prominence, no literal
source tags), but requires descriptors to appear in the first response's text.
Both prompt files are validated before loading the model.

Relative report paths resolve beside the script, so the example writes
`reports/moss-review.html` in this directory. Absolute paths also work, and a
missing `.html` extension is appended. Input, custom prompt, and model paths
resolve from your current working directory.

Other options: `--limit N` for a sample, `--max-seconds N` to cap audio analysis,
`--device auto|cuda|mps|cpu`, and `--model ID_OR_PATH`. By default all files and
the full recording are analyzed. Use `python moss.py --help` for details.

## Pipeline use

Import `load_prompt()`, `load_runtime()`, and `generate_annotation()` from
`moss.py`. Load the model once per job, then reuse it across files.
`generate_annotation()` returns a Python dictionary; the pipeline owns its
persistence. `html_report.py` is loaded only when the CLI report flag is used.
The CLI without that flag prints progress but does not export annotation files.
To customize recovery through the Python API, load its file with `load_prompt(path)`
and pass that text as `recovery_prompt=` to `generate_annotation()`.

## Appendix

There is a leftover directory called `tests` that could provide good initial ideas for pipeline test should this script be intergrated into the pipeline.
