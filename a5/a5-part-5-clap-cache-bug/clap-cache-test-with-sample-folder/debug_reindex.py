"""Run the cache reproduction on copies of the VimSketch samples, outside CI.

The pipeline indexes real WAV files. Voice embeddings are fixed stand-ins, and fake CLAP
provides deterministic text scores. This checks cache behavior, not retrieval quality.
"""

import argparse
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import numpy as np

from vocalocate_demo import app, search, text
from vocalocate_demo.text import ClapScorer
from vocalocate_pipeline import warehouse
from vocalocate_pipeline.config import PipelineConfig


SAMPLES = Path(__file__).resolve().parents[1] / "sample-ref-sfx-and-vocal-im" / "reference-sfx"
HINT = "metallic ringing"


def display_sound(path):
    return " ".join(path.stem.split("_")[-2:]) + f" ({path.name})"


def run_scenario(name, before, after, samples):
    print(f"\n{name}: {len(before)} -> {len(after)} files", flush=True)
    # Copy real SFX into a temporary library; the original WAVs stay untouched.
    with tempfile.TemporaryDirectory(prefix="vocalocate-cache-") as folder:
        root = Path(folder) / "library"
        root.mkdir()
        for i in before:
            shutil.copy2(samples[i], root / samples[i].name)
        print(f"  Original samples folder: {SAMPLES}", flush=True)
        print(f"\n  Temporary library folder: {root}", flush=True)
        print(f"\n  Text hint: {HINT}", flush=True)
        print("\n  Starting sounds:", flush=True)
        for i in before:
            print(f"    - {display_sound(samples[i])}", flush=True)
        for i in before:
            if i not in after:
                print(f"\n  Remove: {display_sound(samples[i])}", flush=True)
        for i in after:
            if i not in before:
                print(f"\n  Add: {display_sound(samples[i])}", flush=True)
        # Fake CLAP generates deterministic stand-in text and SFX audio embeddings.
        state = app.AppState(PipelineConfig(lake_root=Path(folder) / "lake", workers=1),
                             "sfx_meta", clap=ClapScorer(backend="fake"))
        library = app.Library("demo", [str(root)])
        state.libraries[library.name] = library
        state.models["metadata"] = (None, None, "debug-model")
        last_scored_audio = None
        cached_audio_embeddings = text.cached_audio_embeddings

        def record_scored_audio(*args, **kwargs):
            # Save the embedding rows behind the latest score calculation. A stale score hit skips this.
            nonlocal last_scored_audio
            last_scored_audio = cached_audio_embeddings(*args, **kwargs)
            return last_scored_audio

        def index_and_search(order=None):
            state._index(library)
            if library.state != "ready":
                raise RuntimeError(library.message)
            if order is not None:
                # The real indexer orders by path; permute an index to isolate an ordering change.
                old = library.indexes["metadata"]
                paths_by_name = {Path(path).name: i for i, path in enumerate(old.paths)}
                positions = [paths_by_name[samples[i].name] for i in order]
                library.indexes["metadata"] = app.LibraryIndex(
                    [old.clip_ids[i] for i in positions], [old.paths[i] for i in positions],
                    [old.projects[i] for i in positions], old.vectors[positions], old.n_embedded,
                )
            return state.search("demo", "metadata", b"stand-in imitation", HINT, k=20)

        # embed_query returns a fixed stand-in imitation embedding; no imitation WAV is read.
        # search.embed returns fixed stand-in voice embeddings for the indexed reference SFX.
        with patch.object(state, "available_models", return_value=["metadata"]), \
             patch.object(app, "embed_query", return_value=np.array([1.0], dtype=np.float32)), \
             patch.object(text, "cached_audio_embeddings", side_effect=record_scored_audio), \
             patch.object(search, "embed", side_effect=lambda model, feats, ids, *args: np.ones((len(ids), 1))):
            if name == "Reorder":
                print("\n  Simulated index order (the app normally sorts by file path):", flush=True)
                print("\n  Before:", flush=True)
                for i in before:
                    print(f"    - {display_sound(samples[i])}", flush=True)
                print("\n  After:", flush=True)
                for i in after:
                    print(f"    - {display_sound(samples[i])}", flush=True)
            first = index_and_search(before if name == "Reorder" else None)
            assert first["searched"] == len(before)
            again = state.search("demo", "metadata", b"stand-in imitation", HINT, k=20)
            assert "text scores" in again["reused"]
            # Add, remove, or replace copied WAVs, then run the real indexing pipeline again.
            for i in set(before) - set(after):
                (root / samples[i].name).unlink()
            for i in set(after) - set(before):
                shutil.copy2(samples[i], root / samples[i].name)
            # A size mismatch reports current voice scores first, then stale cached text scores.
            # Equal-size replacement can instead reuse stale scores without a NumPy error.
            print(flush=True)  # Separate the scenario details from any error traceback.
            result = index_and_search(after if name == "Reorder" else None)
            assert result["searched"] == len(after)
            # Bypass both caches and score the current cleaned files in index order.
            index = library.indexes["metadata"]
            clean_paths = dict(warehouse.query(state.cfg, "SELECT clip_id, clean_path FROM clips WHERE status = 'ok'"))
            fresh_audio = state.clap.embed_audio([clean_paths[clip_id] for clip_id in index.clip_ids])
            fresh_scores = fresh_audio @ state.clap.embed_text([HINT])[0]
            returned_scores = {hit["path"]: hit["text_score"] for hit in result["results"]}
            assert set(returned_scores) == set(index.paths), "Search results do not match the indexed files"
            assert last_scored_audio is not None and last_scored_audio.shape == fresh_audio.shape, (
                "CLAP embedding rows do not match the current index"
            )
            print("  Fake CLAP per file (embedding check, returned vs fresh score):", flush=True)
            mismatched = []
            for path, scored_audio, current_audio, fresh_score in zip(
                index.paths, last_scored_audio, fresh_audio, fresh_scores
            ):
                same_embedding = np.allclose(scored_audio, current_audio)
                returned = returned_scores[path]
                filename = Path(path).name
                print(f"    {filename}: embedding {'matches' if same_embedding else 'differs'}, "
                      f"score {returned:.5f} vs {fresh_score:.5f}", flush=True)
                if not same_embedding or not np.isclose(returned, fresh_score):
                    mismatched.append(filename)
            assert not mismatched, f"CLAP embedding or score mismatch in {len(mismatched)}/{len(index.paths)} files"
            assert "text scores" not in result["reused"], "Search reused CLAP scores after the index changed"
            print("  PASS: fresh scores match the current files", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("add", "remove", "replace", "reorder", "all"), default="all")
    args = parser.parse_args()
    samples = sorted(SAMPLES.glob("*.wav"))
    if len(samples) != 5:
        raise SystemExit(f"Expected five sample WAVs in {SAMPLES}; found {len(samples)}")
    # Identify the replacement by filename rather than depending on file order.
    dog = next((i for i, p in enumerate(samples) if p.stem.endswith("_Dog_Bark")), None)
    bell = next((i for i, p in enumerate(samples) if p.stem.endswith("_Church bell")), None)
    if dog is None or bell is None:
        raise SystemExit("Expected the dog bark and church bell sample WAVs")
    thunder = next((i for i, p in enumerate(samples) if p.stem.endswith("_Thunder")), None)
    water = next((i for i, p in enumerate(samples) if p.stem.endswith("_Waterfall")), None)
    door = next((i for i, p in enumerate(samples) if p.stem.endswith("_Door_Slam")), None)
    if any(i is None for i in (thunder, water, door)):
        raise SystemExit("Expected thunder, waterfall and door slam sample WAVs")
    baseline = (thunder, bell, water, door)
    without_bell = tuple(i for i in baseline if i != bell)
    for name, before, after in [
        ("Add", baseline, baseline + (dog,)),
        ("Remove", baseline, without_bell),
        ("Replace", baseline, without_bell + (dog,)),
        ("Reorder", baseline, (door, water, thunder, bell)),
    ]:
        if args.scenario in ("all", name.lower()):
            run_scenario(name, before, after, samples)


if __name__ == "__main__":
    main()
