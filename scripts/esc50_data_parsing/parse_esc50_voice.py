"""Write ESC-50-Voice pairs to CSV, sorted by numeric ESC-50 source file ID.

Voice paths are relative to the speaker-folder root; SFX paths to the ESC-50 root.
Assumes each imitation has a corresponding ESC-50 target; audio is not changed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = ("voice_file", "sfx_file", "imitator_id", "category", "recording_id")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "raw_paired_manifest.csv"


def parse_recording(path: Path, voice_root: Path) -> dict[str, str]:
    """Parse SpeakerID/FileName_SpeakerID.wav relative to voice_root."""
    relative = path.relative_to(voice_root)
    speaker_id = relative.parent.name
    suffix = f"_{speaker_id}"
    if not path.stem.endswith(suffix):
        raise ValueError(f"{relative}: filename must end with {suffix}.wav to match its speaker folder")

    target = path.stem[: -len(suffix)]
    # Accept both target_SpeakerID.wav and target.wav_SpeakerID.wav.
    if target.lower().endswith(".wav"):
        target = target[:-4]
    if not target:
        raise ValueError(f"{relative}: missing ESC-50 target filename")

    return {
        "voice_file": relative.as_posix(),
        "sfx_file": f"audio/{target}.wav",
        "imitator_id": f"esc50_voice:{speaker_id}",
        "category": "",
        "recording_id": f"freesound:{int(target.split('-')[1])}",
    }


def add_metadata(rows: list[dict[str, str]], path: Path) -> None:
    """Join ESC-50 labels and source IDs by exact filename, preserving pair rows."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"filename", "category", "src_file"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected filename, category and src_file columns")
        metadata = {row["filename"]: row for row in reader}
    missing = {Path(row["sfx_file"]).name for row in rows} - metadata.keys()
    if missing:
        raise ValueError(f"{len(missing)} targets absent from {path}, e.g. {min(missing)}")
    for row in rows:
        target = metadata[Path(row["sfx_file"]).name]
        row["category"] = target["category"].strip()
        row["recording_id"] = f"freesound:{int(target['src_file'])}"


def build_manifest(root: Path) -> tuple[Path, list[dict[str, str]]]:
    """Accept speaker folders directly, or a parent with vocal_imitation(s)/."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Folder does not exist: {root}. Extract the downloaded archive first.")
    voice_root = next((root / name for name in ("vocal_imitations", "vocal_imitation")
                       if (root / name).is_dir()), root)
    recordings = [
        path for path in voice_root.glob("*/*")
        if path.is_file() and path.suffix.lower() == ".wav"
        and not any(part.startswith(".") or part == "__MACOSX"
                    for part in path.relative_to(voice_root).parts)
    ]
    if not recordings:
        raise ValueError(f"No WAV files found under {voice_root}. Point to the extracted vocal_imitation folder.")

    rows = [parse_recording(path, voice_root) for path in recordings]
    # ESC-50 names: fold-source_id-take-category.wav. Keep speakers of a target together.
    rows.sort(key=lambda row: (
        int(row["recording_id"].split(":", 1)[1]),
        row["sfx_file"],
        int(row["imitator_id"].split(":", 1)[1]),
    ))
    return voice_root, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, type=Path, help="Voice folder containing speaker directories, or its immediate parent")
    parser.add_argument("--esc50-path", type=Path, help="ESC-50 root or audio folder; joins meta/esc50.csv")
    parser.add_argument("--esc50-metadata", type=Path, help="Optional explicit esc50.csv path; also works without ESC-50 audio")
    parser.add_argument("--out", default=DEFAULT_OUTPUT, type=Path,
                        help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args(argv)
    try:
        voice_root, rows = build_manifest(args.path)
        metadata_path = args.esc50_metadata
        if args.esc50_path is not None:
            sfx_root = args.esc50_path.expanduser().resolve()
            if not sfx_root.is_dir():
                raise ValueError(f"ESC-50 folder does not exist: {sfx_root}")
            dataset_root = sfx_root.parent if sfx_root.name == "audio" else sfx_root
            metadata_path = metadata_path or dataset_root / "meta" / "esc50.csv"
        if metadata_path is not None:
            add_metadata(rows, metadata_path.expanduser().resolve())
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} recordings to {output}")
    print(f"Speakers: {len({row['imitator_id'] for row in rows})}")
    print(f"ESC-50 targets: {len({row['sfx_file'] for row in rows})}")
    print(f"Read voice files from: {voice_root}")
    print("Voice paths are relative to the voice folder; SFX paths to the ESC-50 root.")
    if metadata_path is None:
        print("Categories left empty; supply --esc50-path or --esc50-metadata to add them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
