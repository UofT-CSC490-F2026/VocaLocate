"""Team imitation recordings (A2 Part Two, "Data we can generate").

Layout:
    recordings.csv with columns
        file, imitator_id, target_group_key, text_hint, hint_type
    and optionally split, query_id; plus the audio files it lists (paths relative to the CSV).
`target_group_key` names the reference sound being imitated in another
source, e.g. "esc50:100032" or "vimsketch:<reference name>". Rows with a text
hint become voice, text and target triples in the queries table (`query_id`, if
given, names the query; otherwise it is a hash of the file and the hint).

Team recordings are never used for training. Part Two splits the target sounds
before recording into a validation group (to tune the fusion weight) and a test
group: `split` is val or test, and every recording of a target gets the target's
split. A target with no split in the CSV gets a seeded 50/50 hash of its group key
(splits.held_out_split).
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from . import IngestResult, RawFile

SOURCE = "team"
VERSION = "live"
LICENCE = "team-internal"


_SPLITS = {"val": "val", "validation": "val", "test": "test"}


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def _split(row: dict) -> str | None:
    value = _clean(row.get("split"))
    if value is None:
        return None
    if value.lower() not in _SPLITS:
        raise ValueError(f"{row['file']}: split must be val or test (team recordings are never trained on), "
                         f"got {value!r}")
    return _SPLITS[value.lower()]


def target_splits(rows: list[dict]) -> dict[str, str]:
    """The split of each target named in the CSV. Two different splits for one target is an error."""
    out: dict[str, str] = {}
    for row in rows:
        split, target = _split(row), row["target_group_key"].strip()
        if split is not None and out.setdefault(target, split) != split:
            raise ValueError(f"target {target!r} is in both {out[target]!r} and {split!r}")
    return out


def ingest(root: Path) -> IngestResult:
    meta = root / "recordings.csv"
    if not meta.exists():
        raise FileNotFoundError(f"Team recordings manifest not found at {meta}")
    with open(meta, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    split_of = target_splits(rows)
    files, queries, splits, missing = [], [], {}, []
    for row in rows:
        path = root / row["file"]
        if not path.exists():
            missing.append(row["file"])
            continue
        target = row["target_group_key"].strip()
        files.append(
            RawFile(
                source=SOURCE,
                source_version=VERSION,
                source_key=row["file"],
                path=path,
                role="imitation",
                group_key=target,
                imitator_id=row["imitator_id"].strip(),
                licence=LICENCE,
            )
        )
        if target in split_of:
            splits[row["file"]] = split_of[target]
        hint = (row.get("text_hint") or "").strip()
        if hint:
            queries.append(
                {
                    "query_id": _clean(row.get("query_id"))
                    or hashlib.sha256(f"{row['file']}|{hint}".encode()).hexdigest()[:32],
                    "source_key": row["file"],
                    "text_hint": hint,
                    "hint_type": (row.get("hint_type") or "").strip() or None,
                    "target_group_key": target,
                }
            )
    return IngestResult(files=files, queries=queries, splits=splits, missing=missing)
