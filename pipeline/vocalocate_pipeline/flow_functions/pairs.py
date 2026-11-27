"""Training pairs: link each clean imitation to the reference sound it imitates."""

from __future__ import annotations

from collections.abc import Iterable


def build_pairs(clips: Iterable[dict], targets: dict[str, str] | None = None) -> list[dict]:
    """Pair each clean imitation with the clean reference of the same group in the same source.

    `targets` maps an imitation's source key to its reference's, for groups that hold several
    references (IngestResult.targets); such an imitation pairs only with that reference.
    """
    targets = targets or {}
    clips = [c for c in clips if c["status"] == "ok" and c["group_key"]]
    refs = {(c["source"], c["group_key"]): c["clip_id"] for c in clips if c["role"] == "reference"}
    ref_by_key = {(c["source"], c["source_key"]): c for c in clips if c["role"] == "reference"}
    pairs = []
    for c in clips:
        if c["role"] != "imitation":
            continue
        if c["source_key"] in targets:
            target = ref_by_key.get((c["source"], targets[c["source_key"]]))
            ref = target["clip_id"] if target is not None and target["group_key"] == c["group_key"] else None
        else:
            ref = refs.get((c["source"], c["group_key"]))
        if ref is not None:
            pairs.append(
                {
                    "source": c["source"],
                    "imitation_clip_id": c["clip_id"],
                    "reference_clip_id": ref,
                    "group_key": c["group_key"],
                }
            )
    return pairs
