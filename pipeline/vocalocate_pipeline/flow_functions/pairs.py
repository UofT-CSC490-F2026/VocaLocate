"""Training pairs: link each clean imitation to the reference sound it imitates."""

from __future__ import annotations

from collections.abc import Iterable


def build_pairs(clips: Iterable[dict]) -> list[dict]:
    """Pair each clean imitation with the clean reference of the same group in the same source."""
    clips = [c for c in clips if c["status"] == "ok" and c["group_key"]]
    refs = {(c["source"], c["group_key"]): c["clip_id"] for c in clips if c["role"] == "reference"}
    pairs = []
    for c in clips:
        if c["role"] != "imitation":
            continue
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
