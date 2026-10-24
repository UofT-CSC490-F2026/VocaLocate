"""Team imitation recordings (A2 Part Two, "Data we can generate").

Layout:
    recordings.csv with columns
        file, imitator_id, target_group_key, text_hint, hint_type
    and optionally split, query_id, vocalization_type, device_type; plus the audio files
    it lists (paths relative to the CSV).
    imitators.csv (optional) with columns
        imitator_id, role, years_experience, first_language, music_training, consent_scope
    judgements.csv (optional) with columns
        query_id, candidate_file or candidate_clip_id, relevance, rater_id
`target_group_key` names the reference sound being imitated in another
source, e.g. "esc50:100032" or "vimsketch:<reference name>". Rows with a text
hint become voice, text and target triples in the queries table (`query_id`, if
given, names the query; otherwise it is a hash of the file and the hint).

Team recordings are never used for training. Part Two splits the target sounds
before recording into a validation group (to tune the fusion weight) and a test
group: `split` is val or test, and every recording of a target gets the target's
split. A target with no split in the CSV gets a seeded 50/50 hash of its group key
(splits.held_out_split).

`vocalization_type` is free text (hum, whistle, beatbox, ...); `device_type` is
computer mic, headset mic, studio mic or other. Both go to imitation_metadata.

A judgement rates how well a candidate SFX answers a query, from 0 (not relevant)
to 3 (the sound wanted). `query_id` is a query's ID as v_retrieval_query shows it;
the candidate is either a file (relative to the CSV, identified by its hash like
every clip) or a clip_id.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from ..audio import sha256_file
from ..schemas import SchemaError
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


def _device(row: dict) -> str | None:
    value = _clean(row.get("device_type"))
    return value.lower().replace("-", "_").replace(" ", "_") if value else None


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
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def read_judgements(root: Path, missing: list[str]) -> list[dict]:
    out = []
    for row in _read(root / "judgements.csv"):
        candidate = _clean(row.get("candidate_clip_id"))
        file = _clean(row.get("candidate_file"))
        if candidate is None and file is not None:
            if not (root / file).exists():
                missing.append(file)
                continue
            candidate = sha256_file(root / file)
        try:
            relevance = int(row["relevance"].strip())
        except (KeyError, AttributeError, ValueError):
            got = row.get("relevance")
            raise SchemaError(f"judgements.csv: relevance must be an integer 0-3, got {got!r}") from None
        out.append(
            {
                "query_id": row["query_id"].strip(),
                "candidate_clip_id": candidate,
                "relevance": relevance,
                "rater_id": _clean(row.get("rater_id")),
            }
        )
    return out


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
    files, queries, metadata, splits, missing = [], [], [], {}, []
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
        vocalization, device = _clean(row.get("vocalization_type")), _device(row)
        if vocalization or device:
            metadata.append({"source_key": row["file"], "vocalization_type": vocalization, "device_type": device})
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
    return IngestResult(
        files=files,
        queries=queries,
        imitation_metadata=metadata,
        imitators=read_imitators(root),
        judgements=read_judgements(root, missing),
        splits=splits,
        missing=missing,
    )
