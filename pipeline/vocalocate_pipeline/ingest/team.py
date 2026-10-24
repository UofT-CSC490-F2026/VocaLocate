"""Team imitation recordings (A2 Part Two, "Data we can generate").

Layout:
    recordings.csv with columns
        file, imitator_id, target_group_key, text_hint, hint_type
    and optionally split; plus the audio files it lists (paths relative to the CSV).
    imitators.csv (optional) with columns
        imitator_id, role, years_experience, first_language, music_training, consent_scope
`target_group_key` names the reference sound being imitated in another
source, e.g. "esc50:100032" or "vimsketch:<reference name>". Rows with a text
hint become voice, text and target triples in the queries table (the query ID
is a hash of the file and the hint).

Team recordings are never used for training. Part Two splits the target sounds
before recording into a validation group (to tune the fusion weight) and a test
group: `split` is val or test, and every recording of a target gets the target's
split. A target with no split in the CSV gets a seeded 50/50 hash of its group key
(flow_functions.splits.held_out_split).
"""

from __future__ import annotations

from pathlib import Path

from . import IngestBuilder, IngestResult, read_table, require

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


def _bool(value: str | None) -> bool | None:
    value = (_clean(value) or "").lower()
    if not value:
        return None
    if value in {"true", "yes", "y", "1", "t"}:
        return True
    if value in {"false", "no", "n", "0", "f"}:
        return False
    raise ValueError(f"not a yes/no value: {value!r}")


def _int(value: str | None) -> int | None:
    value = _clean(value)
    return int(value) if value is not None else None


def _read(path: Path) -> list[dict]:
    """An optional team CSV: no rows when the file is absent."""
    return read_table(path) if path.exists() else []


def read_imitators(root: Path) -> list[dict]:
    return [
        {
            "imitator_id": row["imitator_id"].strip(),
            "role": _clean(row.get("role")),
            "years_experience": _int(row.get("years_experience")),
            "first_language": _clean(row.get("first_language")),
            "music_training": _bool(row.get("music_training")),
            "consent_scope": _clean(row.get("consent_scope")),
        }
        for row in _read(root / "imitators.csv")
    ]


def target_splits(rows: list[dict]) -> dict[str, str]:
    """The split of each target named in the CSV. Two different splits for one target is an error."""
    out: dict[str, str] = {}
    for row in rows:
        split, target = _split(row), row["target_group_key"].strip()
        if split is not None and out.setdefault(target, split) != split:
            raise ValueError(f"target {target!r} is in both {out[target]!r} and {split!r}")
    return out


def ingest(root: Path) -> IngestResult:
    rows = read_table(require(root / "recordings.csv", "Team recordings manifest"))
    split_of = target_splits(rows)
    b = IngestBuilder(root, SOURCE, VERSION, LICENCE)
    for row in rows:
        key, target = row["file"], row["target_group_key"]
        if not b.add_file(key, "imitation", group_key=target, imitator_id=row["imitator_id"]):
            continue
        if target in split_of:
            b.set_split(key, split_of[target])
        if row.get("text_hint"):
            b.add_query(key, row["text_hint"], _clean(row.get("hint_type")), target)
    b.imitators = read_imitators(root)
    return b.result()
