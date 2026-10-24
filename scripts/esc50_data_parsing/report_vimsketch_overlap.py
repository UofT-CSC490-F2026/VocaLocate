"""List ESC, VimSketch and VIS reference filenames sharing a Freesound upload.

Uses metadata and filename lists only. Shared IDs establish common origin,
not identical audio or duplicate vocal imitations. No input files are modified.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
FIELDS = ("recording_id", "esc50_filename", "vimsketch_filename", "vis_filename")


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected columns {', '.join(sorted(required))}")
        return list(reader)


def original_name(filename: str) -> str:
    """Remove VimSketch's combined-dataset number, retaining the original name."""
    match = re.fullmatch(r"(\d+)_(.+\.wav)", filename, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid VimSketch filename: {filename}")
    return match[2]


def name_key(filename: str) -> str:
    """Account for VIS/VimSketch differences in punctuation and _reference suffix."""
    stem = re.sub(r"_reference$", "", Path(filename).stem, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def reference_names(root: Path) -> list[str]:
    path = root / "reference_file_names.csv"
    if path.is_file():
        with path.open(newline="", encoding="utf-8-sig") as stream:
            names = [row[0].strip() for row in csv.reader(stream) if row and row[0].strip()]
    else:
        directory = root / "references"
        if not directory.is_dir():
            raise ValueError(f"Need {path} or {directory}")
        names = [p.name for p in directory.iterdir()
                 if p.is_file() and p.suffix.lower() == ".wav" and not p.name.startswith(".")]
    if not names:
        raise ValueError(f"No reference filenames found under {root}")
    return sorted(set(names))


def build_report(manifest: Path, vis_metadata: Path, vimsketch: Path):
    pairs = read_csv(manifest, {"sfx_file", "recording_id"})
    licence = read_csv(vis_metadata, {"filename", "freesound_id"})
    metadata: dict[str, set[tuple[str, str]]] = {}
    for row in licence:
        filename, fid = row["filename"].strip(), row["freesound_id"].strip()
        if fid and not re.fullmatch(r"\d+", fid):
            raise ValueError(f"Invalid Freesound ID for {filename}: {fid}")
        metadata.setdefault(name_key(filename), set()).add((fid, filename))

    references = reference_names(vimsketch)
    # The release defines references 001-302 as VIS; 303-542 are VocalSketch.
    vis_refs = []
    for name in references:
        original_name(name)  # Validate before selecting by the combined reference number.
        if 1 <= int(name.split("_", 1)[0]) <= 302:
            vis_refs.append(name)
    if not vis_refs:
        raise ValueError("No VIS references (VimSketch reference numbers 001-302) found")

    by_recording: dict[str, list[dict[str, str]]] = {}
    warnings = []
    mapped = 0
    for name in vis_refs:
        original = original_name(name)
        matches = metadata.get(name_key(original), set())
        if len(matches) > 1:
            raise ValueError(f"Ambiguous VIS metadata match for {name}: {sorted(matches)}")
        if not matches:
            warnings.append(f"No VIS metadata match for {name}; overlap is unknown")
            continue
        fid, vis_filename = next(iter(matches))
        if not fid:
            warnings.append(f"No Freesound ID for {name}; overlap is unknown")
            continue
        mapped += 1
        by_recording.setdefault(f"freesound:{int(fid)}", []).append({
            "vimsketch_filename": name,
            "vis_filename": vis_filename,
        })

    targets = set()
    for row in pairs:
        if not row["sfx_file"] or not row["recording_id"]:
            raise ValueError("Manifest rows need sfx_file and recording_id values")
        if not re.fullmatch(r"freesound:\d+", row["recording_id"]):
            raise ValueError(f"Invalid manifest recording_id: {row['recording_id']}")
        recording_id = f"freesound:{int(row['recording_id'].split(':')[1])}"
        targets.add((recording_id, Path(row["sfx_file"]).name))

    report = []
    for recording_id, filename in targets:
        for match in by_recording.get(recording_id, []):
            report.append({"recording_id": recording_id, "esc50_filename": filename, **match})
    report.sort(key=lambda row: (int(row["recording_id"].split(":")[1]),
                                 row["esc50_filename"], row["vimsketch_filename"]))
    return report, warnings, mapped, len(vis_refs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=OUTPUT_DIR / "raw_paired_manifest.csv",
                        help="ESC paired CSV (default: output/raw_paired_manifest.csv beside this script)")
    parser.add_argument("--vis-metadata", required=True, type=Path,
                        help="VIS original_recordings_license.csv containing filename and freesound_id")
    parser.add_argument("--vimsketch-path", required=True, type=Path,
                        help="VimSketch root containing reference_file_names.csv or references/")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "vimsketch_vis_overlaps.csv",
                        help="Report CSV (default: output/vimsketch_vis_overlaps.csv beside this script)")
    args = parser.parse_args(argv)
    manifest, metadata, vimsketch, output = (
        path.expanduser().resolve()
        for path in (args.manifest, args.vis_metadata, args.vimsketch_path, args.out)
    )
    if output in {manifest, metadata, vimsketch / "reference_file_names.csv"}:
        parser.error("Choose a report output separate from the input files")
    try:
        report, warnings, mapped, total = build_report(manifest, metadata, vimsketch)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(report)
    except (ValueError, OSError, csv.Error) as exc:
        parser.error(str(exc))
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    print(f"Checked Freesound IDs for {mapped}/{total} VimSketch VIS references.")
    print(f"Shared Freesound uploads: {len({row['recording_id'] for row in report})}")
    print(f"Affected ESC targets: {len({row['esc50_filename'] for row in report})}")
    print(f"Wrote {len(report)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
