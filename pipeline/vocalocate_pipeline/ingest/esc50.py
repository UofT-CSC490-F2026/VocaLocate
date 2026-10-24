"""ESC-50 (github.com/karolpiczak/ESC-50): 2,000 five-second clips in 50 classes.

Layout: meta/esc50.csv (filename, fold, target, category, esc10, src_file, take)
and audio/<filename>. We use it as a small stand-in library to test the
pipeline end to end, not for training.
"""

from __future__ import annotations

from pathlib import Path

from . import IngestBuilder, IngestResult, read_table, require

SOURCE = "esc50"
VERSION = "2.0"
LICENCE = "CC BY-NC 3.0"


def ingest(root: Path) -> IngestResult:
    b = IngestBuilder(root, SOURCE, VERSION, LICENCE)
    for row in read_table(require(root / "meta" / "esc50.csv", "ESC-50 metadata")):
        b.add_file(
            f"audio/{row['filename']}",
            "library",
            label=row["category"],
            # Takes cut from the same Freesound recording share a group,
            # so they can never be split across train and test.
            group_key=f"{SOURCE}:{row['src_file']}",
            extra={"fold": int(row["fold"]), "esc10": row["esc10"] == "True", "take": row["take"]},
        )
    return b.result()
