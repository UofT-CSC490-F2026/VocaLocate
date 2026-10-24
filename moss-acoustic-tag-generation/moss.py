#!/usr/bin/env python3
"""Generate acoustic annotations with MOSS-Audio-4B-Instruct.
The main prompt asks MOSS to generate acoustic tags from audio; Python parses
those generated tags. If none are parsed, a text-only fallback attempts to
extract descriptors from the first response.
Input audio folders are searched recursively. Add --html-report PATH for a
human-review report; otherwise no report is written. Pipeline code can import
the inference functions directly.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Identifies the MOSS model weights and configurations to pull from HF
MODEL_ID = "OpenMOSS-Team/MOSS-Audio-4B-Instruct"

# TODO: Align file extension support with what's already being processed in pipeline
EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}

DEFAULT_PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "audio_tag_extraction_instruction.txt"
DEFAULT_RECOVERY_PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "tag_extraction.txt"


def load_prompt(path: Path | str | None = None, *, text: str | None = None) -> str:
    """Read one complete prompt, including task and output instructions.

    Explicit relative paths resolve from the working directory. The bundled
    default resolves beside this script. Headings are sent to MOSS as text;
    no section splitting or extra output instructions are added.
    """
    if path is not None and text is not None:
        raise ValueError("Choose a prompt file or inline prompt, not both")
    if text is None:
        prompt_path = Path(path).expanduser() if path is not None else DEFAULT_PROMPT_FILE
        text = prompt_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Prompt must not be empty")
    return text.strip()


def parse_annotation(raw: str) -> dict:
    """Accept inline/Markdown labels and retain prose instead of blanking the row."""
    # Labels may be inline, reordered, Markdown headings, or bullet items.
    pattern = r"(?<!\w)(?:\*\*)?(description|(?:acoustic\s+)?tags|temporal[ _]pattern|raw[ _]model[ _]output)(?:\*\*)?\s*:(?:\*\*)?\s*"
    matches = list(re.finditer(pattern, raw, flags=re.IGNORECASE))
    fields = {}
    for index, match in enumerate(matches):
        key = match[1].lower().replace(" ", "_").removeprefix("acoustic_")
        if key == "raw_model_output":
            # This label ends the annotation fields. The actual model response
            # is already captured by annotate_audio(), including this section.
            break
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        fields.setdefault(key, raw[match.end():end].strip().rstrip("* \n"))
    if not fields.get("description"):
        fields["description"] = raw[:matches[0].start()].strip() if matches else raw.strip()
    tag_text = fields.get("tags", "")
    tags = split_tags(tag_text)
    valid = (bool(fields.get("description")) and 1 <= len(tags) <= 5
             and all(len(tag.split()) <= 6 for tag in tags))
    result = dict(description=fields.get("description", ""), tags=tags,
                  temporal_pattern=fields.get("temporal_pattern", "unknown"),
                  status="success" if valid else "fail")
    if not valid:
        result["error"] = "No usable tag list (expected 1–5 short acoustic descriptors); original prose retained."
    return result


def split_tags(text: str) -> list[str]:
    """Clean explicit comma, semicolon, or bullet lists without inventing tags."""
    tags = []
    for item in re.split(r"[,;\n]", text):
        item = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", item)
        item = item.strip(" \t\r\n.\"'`*[]").lower().replace("_", " ")
        if item and item not in {"unknown", "none", "n/a"} and item not in tags:
            tags.append(item)
    return tags


def collect_audio(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"Not an input folder: {folder}")
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def select_device(torch, requested: str) -> str:
    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
    if requested == "auto":
        return next(name for name, usable in available.items() if usable)
    if not available[requested]:
        raise ValueError(f"Requested device is unavailable: {requested}")
    return requested

def huggingface_auth_kwargs() -> dict[str, str]:
    """Read credentials injected by a shell or pipeline's secret manager.

    Leave authentication unspecified when HF_TOKEN is absent so public/local
    models and an existing `hf auth login` session continue to work. Do not
    print this dictionary or add it to annotation/report metadata.
    """
    token = os.environ.get("HF_TOKEN", "").strip()
    return {"token": token} if token else {}


def load_runtime(model_id: str, requested_device: str):
    """Import the installed MOSS package and download/reuse cached HF weights."""
    import torch
    try:
        from src.modeling_moss_audio import MossAudioModel
        from src.processing_moss_audio import MossAudioProcessor
    except ImportError as exc:
        raise ImportError(
            "MOSS-Audio or its runtime dependencies are unavailable. Activate your "
            "MOSS environment and install the MOSS-Audio package; see README.md. "
            f"Original import error: {exc}"
        ) from exc

    device = select_device(torch, requested_device)
    # Pipeline authentication step: inject HF_TOKEN into the job environment.
    auth_kwargs = huggingface_auth_kwargs()
    print(f"Loading {model_id} on {device}. First use downloads the model.", flush=True)
    model = MossAudioModel.from_pretrained(
        model_id, trust_remote_code=True,
        dtype=torch.float32 if device == "cpu" else "auto", device_map=device,
        **auth_kwargs,
    ).eval()
    processor = MossAudioProcessor.from_pretrained(
        model_id, trust_remote_code=True, enable_time_marker=True,
        **auth_kwargs,
    )
    return model, processor, torch


def generate_response(prompt, model, processor, torch, audio=None, max_new_tokens=320):
    """Use MOSS's native audio tensors, dtype and token mask."""
    kwargs = {"text": prompt, "return_tensors": "pt"}
    if audio is not None:
        kwargs["audios"] = [audio]
    inputs = processor(**kwargs).to(model.device)
    if inputs.get("audio_data") is not None:
        inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
    inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        )
    input_length = inputs["input_ids"].shape[1]
    return processor.decode(output[0, input_length:], skip_special_tokens=True).strip()


def annotate_audio(audio, prompt, model, processor, torch, *, recovery_prompt: str | None = None) -> dict:
    """Analyze audio, then recover tags from text only if the parser found none.

    In the first pass, the supplied prompt asks MOSS to generate acoustic tags
    directly from the recording. Python then extracts those already-generated
    tags from the response's labeled fields.

    If MOSS returns text without a parseable tag list, the fallback receives
    only that text response. It salvages acoustic descriptors
    already written there; it does not listen again or correct acoustic mistakes.
    Responses with parsed tags return immediately without a second model call.
    Recovery instructions come from the supplied text or prompts/tag_extraction.txt.
    """
    # Pass 1: Ask MOSS for tags from AUDIO, then parse its generated tag fields.
    audio_response = generate_response(prompt, model, processor, torch, audio=audio)
    result = {"raw_model_output": audio_response, **parse_annotation(audio_response)}
    if result["tags"] or not audio_response:
        return result

    # Pass 2 (fallback): RESPONSE TEXT ONLY -> descriptors copied from that text.
    # The main prompt already requested tags. This recovers a missing tag field
    # from the resulting prose; no audio is resent or analyzed in this pass.
    try:
        recovery_instructions = (load_prompt(DEFAULT_RECOVERY_PROMPT_FILE)
                                 if recovery_prompt is None else load_prompt(text=recovery_prompt))
        text_extraction_prompt = recovery_instructions + "\n\nMODEL RESPONSE TO PROCESS\n" + audio_response
        text_extraction_response = generate_response(
            text_extraction_prompt, model, processor, torch, audio=None, max_new_tokens=96,
        )
        result["raw_model_output"] += "\n\nText-only tag extraction:\n" + text_extraction_response
        normalize = lambda value: " ".join(value.casefold().replace("_", " ").split())
        source = normalize(audio_response)
        # Check textual presence only; this does not verify accuracy against audio.
        tags = [tag for tag in split_tags(text_extraction_response)
                if len(tag.split()) <= 6
                and re.search(r"(?<!\w)" + re.escape(normalize(tag)) + r"(?!\w)", source)]
        if tags:
            result.update(tags=tags[:5], status="success")
            result.pop("error", None)
    except (OSError, ValueError, RuntimeError) as exc:
        result.update(status="fail", error=f"Text-only extraction failed: {exc}")
    return result


def generate_annotation(path, prompt, max_seconds, model, processor, torch, *, recovery_prompt: str | None = None) -> dict:
    """Return one annotation dictionary without generating files or importing report code."""
    from src.audio_io import load_audio

    sample_rate = int(processor.config.mel_sr)
    audio = load_audio(str(path), sample_rate=sample_rate)
    full_seconds = len(audio) / sample_rate
    if max_seconds is not None:
        audio = audio[:max(1, int(max_seconds * sample_rate))]
    if len(audio) == 0:
        raise ValueError("Empty audio")
    result = annotate_audio(audio, prompt, model, processor, torch, recovery_prompt=recovery_prompt)
    result.update(duration_sec=round(full_seconds, 3),
                  analyzed_sec=round(len(audio) / sample_rate, 3), sample_rate=sample_rate)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="SFX folder (searched recursively)")
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt", help="Complete inline prompt, including output-format instructions")
    prompts.add_argument("--prompt-file", type=Path,
                         help="Complete UTF-8 prompt file; default: bundled prompts/audio_tag_extraction_instruction.txt")
    parser.add_argument("--recovery-prompt-file", type=Path, default=DEFAULT_RECOVERY_PROMPT_FILE,
                        help="UTF-8 instructions for text-only tag recovery; default: bundled prompts/tag_extraction.txt")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--html-report", type=Path, metavar="PATH",
                        help="Generate an HTML review report at PATH, e.g. reports/review.html; relative paths resolve beside this script")
    # Preserve previous local commands while keeping the current CLI concise.
    output.add_argument("--output-name", dest="html_report", type=Path,
                        help=argparse.SUPPRESS)
    output.add_argument("--no-report", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face ID or local model directory")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Optionally analyze only the first N seconds; default: entire recording")
    parser.add_argument("--limit", type=int, default=0, help="Process first N files; 0 means all")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output")
    args = parser.parse_args(argv)
    if (args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0)) or args.limit < 0:
        parser.error("--max-seconds must be positive and finite; --limit must be nonnegative")
    try:
        folder = args.folder.expanduser().resolve()
        files = collect_audio(folder)
        if args.limit:
            files = files[:args.limit]
        if not files:
            raise ValueError("No supported audio files found")
        prompt = load_prompt(args.prompt_file, text=args.prompt)
        recovery_prompt = load_prompt(args.recovery_prompt_file)
        report = None
        if args.html_report is not None:
            # Human review is optional; pipeline imports do not depend on HTML code.
            from html_report import report_path, write_html_report

            report = report_path(args.html_report)
            if report.exists() and not args.overwrite:
                raise ValueError(f"Report exists: {report}; choose another report path or use --overwrite")
        model_id = str(Path(args.model).expanduser()) if args.model.startswith("~") else args.model
        model, processor, torch = load_runtime(model_id, args.device)
        if report is not None:
            report.parent.mkdir(parents=True, exist_ok=True)
        records = []
        failures = 0
        try:
            for index, path in enumerate(files, 1):
                record = {"file": str(path), "source_key": path.relative_to(folder).as_posix(),
                          "model": model_id, "prompt": prompt, "recovery_prompt": recovery_prompt,
                          "generated_at": datetime.now(timezone.utc).isoformat(),
                          "max_seconds": args.max_seconds}
                try:
                    with path.open("rb") as audio_file:
                        record["file_sha256"] = hashlib.file_digest(audio_file, "sha256").hexdigest()
                    record.update(generate_annotation(
                        path, prompt, args.max_seconds, model, processor, torch,
                        recovery_prompt=recovery_prompt,
                    ))
                except Exception as exc:
                    record.update(status="fail", tags=[], error=f"{type(exc).__name__}: {exc}")
                if report is not None:
                    records.append(record)
                failures += record["status"] != "success"
                print(f"[{index}/{len(files)}] {record['source_key']}: {record['status']} {record['tags']}", flush=True)
                if record.get("error"):
                    print(f"  {record['error']}", flush=True)
        finally:
            # Save completed rows even if a later file is interrupted with Ctrl+C.
            if report is not None:
                write_html_report(records, report)
        if report is not None:
            print(f"Review report: {report}; {failures} file(s) failed.")
        else:
            print(f"Processed {len(files)} file(s); {failures} failed.")
        return 1 if failures else 0
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
