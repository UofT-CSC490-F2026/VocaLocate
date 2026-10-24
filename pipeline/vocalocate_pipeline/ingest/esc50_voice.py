"""ESC-50-Voice (Okamoto et al., ICASSP 2024; zenodo.org/records/11385662): 9,920 vocal imitations.

Eight imitators each imitated the same 1,240 ESC-50 clips (31 of its 50 classes, listed in the
release's EventList.csv). The release holds only the imitations; the ESC-50 clips they imitate
come from ESC-50 itself. Layout, with ESC-50 unpacked inside the folder passed in or next to it:

    vocal_imitation/<speaker>/<ESC-50 filename>_<speaker>.wav     e.g. 000/1-137-A-32_000.wav
    ESC-50/meta/esc50.csv, ESC-50/audio/<ESC-50 filename>         (any folder name works)

Some imitations are named <ESC-50 filename>.wav_<speaker>.wav; both forms are read.

* Each imitated ESC-50 clip is a reference; its category is the ESC-50 class and its
  recording_id is its Freesound upload (src_file), as for the Vocal Imitation Set.
* Several takes cut from one Freesound upload were imitated separately (1,240 clips from 946
  uploads), so a group is an upload and holds one reference per take. Each imitation is paired
  with the exact take it imitates (`IngestResult.targets`), not just any reference of its group.
* The Vocal Imitation Set holds 64 of the same Freesound uploads (3 as references, the rest as
  same-class recordings). Splits are hashed by the Vocal Imitation Set's group key for the upload,
  vis:freesound:<id>, so a shared upload always lands in the same split in both sources. The key
  depends only on the Freesound ID, so this needs no Vocal Imitation Set files.
* The record says every imitator is a Japanese speaker; that is the only profile field it gives.

Licence: free for research use; the authors ask that the data not be redistributed.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import IngestBuilder, IngestResult, read_table, require

SOURCE = "esc50_voice"
VERSION = "1.0"
LICENCE = "Research use only, no redistribution (ESC-50-Voice); ESC-50 clips: CC BY-NC 3.0"
VOICE_FOLDERS = ("vocal_imitation", "vocal_imitations")


def _children(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))) if d.is_dir() else []


def _speaker_dirs(d: Path) -> list[Path]:
    return [p for p in _children(d) if p.name.isdigit()]


def find_dirs(root: Path) -> tuple[Path, Path]:
    """(folder of speaker folders, ESC-50 root), from the voice folder, its parent, or a folder holding both."""
    candidates = [root / name for name in VOICE_FOLDERS] + [root]
    candidates += [d / name for d in _children(root) for name in VOICE_FOLDERS]
    voice = next((d for d in candidates if _speaker_dirs(d)), None)
    if voice is None:
        raise FileNotFoundError(f"ESC-50-Voice speaker folders (vocal_imitation/<speaker>/) not found under {root}")
    release = voice.parent if voice.name in VOICE_FOLDERS else voice
    for d in [release, *_children(release), *_children(release.parent)]:
        if (d / "meta" / "esc50.csv").exists():
            return voice, d
    raise FileNotFoundError(f"ESC-50 (meta/esc50.csv) not found in or next to {root}; it is not part of the release")


def target_name(path: Path) -> str:
    """The ESC-50 filename an imitation imitates, from <target>_<speaker>.wav in folder <speaker>."""
    speaker = path.parent.name
    if not path.stem.endswith(f"_{speaker}"):
        raise ValueError(f"{path}: expected a name ending in _{speaker}.wav, like its speaker folder")
    target = path.stem[: -len(speaker) - 1].removesuffix(".wav")
    if not target:
        raise ValueError(f"{path}: no ESC-50 filename before _{speaker}")
    return f"{target}.wav"


def ingest(root: Path) -> IngestResult:
    root = Path(root)
    voice, esc = find_dirs(root)
    esc_rows = {r["filename"]: r for r in read_table(require(esc / "meta" / "esc50.csv", "ESC-50 metadata"))}

    def rel(path: Path) -> str:
        return Path(os.path.relpath(path, root)).as_posix()

    imitations = sorted(p for d in _speaker_dirs(voice) for p in d.iterdir()
                        if p.is_file() and p.suffix.lower() == ".wav" and not p.name.startswith("."))
    unknown = sorted({target_name(p) for p in imitations} - esc_rows.keys())
    if unknown:
        raise ValueError(f"{len(unknown)} imitated clips are not in {esc / 'meta' / 'esc50.csv'}, e.g. {unknown[0]}")

    b = IngestBuilder(root, SOURCE, VERSION, LICENCE)
    for path in imitations:
        row = esc_rows[target_name(path)]
        recording = f"freesound:{int(row['src_file'])}"
        group = f"{SOURCE}:{recording}"
        ref_path = esc / "audio" / row["filename"]
        ref_key = rel(ref_path)
        if not b.has(ref_key) and b.add_file(ref_key, "reference", path=ref_path, label=row["category"],
                                             group_key=group, extra={"fold": int(row["fold"]), "take": row["take"]}):
            b.add_metadata(ref_key, row["category"], recording_id=recording)
            b.set_split_group(ref_key, f"vis:{recording}")
        speaker = path.parent.name
        key = rel(path)
        b.add_file(key, "imitation", path=path, label=row["category"], group_key=group,
                   imitator_id=f"{SOURCE}:{speaker}")
        b.set_target(key, ref_key)
        b.set_split_group(key, f"vis:{recording}")
    b.imitators = [{"imitator_id": f"{SOURCE}:{d.name}", "first_language": "Japanese"} for d in _speaker_dirs(voice)]
    return b.result()
