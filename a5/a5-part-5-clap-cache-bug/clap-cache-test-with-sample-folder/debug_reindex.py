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

from vocalocate_demo import app, search
from vocalocate_demo.text import ClapScorer
from vocalocate_pipeline.config import PipelineConfig


SAMPLES = Path(__file__).resolve().parents[1] / "sample-ref-sfx-and-vocal-im" / "reference-sfx"
HINT = "dog bark"


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
        print(f"  Temporary library folder: {root}", flush=True)
        print(f"  Text hint: {HINT}", flush=True)
        print("  Starting sounds:", flush=True)
        for i in before:
            print(f"    - {display_sound(samples[i])}", flush=True)
        for i in before:
            if i not in after:
                print(f"  Remove: {display_sound(samples[i])}", flush=True)
        for i in after:
            if i not in before:
                print(f"  Add: {display_sound(samples[i])}", flush=True)
        # Fake CLAP generates deterministic stand-in text and SFX audio embeddings.
        state = app.AppState(PipelineConfig(lake_root=Path(folder) / "lake", workers=1),
                             "sfx_meta", clap=ClapScorer(backend="fake"))
        library = app.Library("demo", [str(root)])
        state.libraries[library.name] = library
        state.models["metadata"] = (None, None, "debug-model")

        def index_and_search():
            state._index(library)
            if library.state != "ready":
                raise RuntimeError(library.message)
            return state.search("demo", "metadata", b"stand-in imitation", HINT, k=20)

        # embed_query returns a fixed stand-in imitation embedding; no imitation WAV is read.
        # search.embed returns fixed stand-in voice embeddings for the indexed reference SFX.
        with patch.object(state, "available_models", return_value=["metadata"]), \
             patch.object(app, "embed_query", return_value=np.array([1.0], dtype=np.float32)), \
             patch.object(search, "embed", side_effect=lambda model, feats, ids, *args: np.ones((len(ids), 1))):
            first = index_and_search()
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
            result = index_and_search()
            assert result["searched"] == len(after)
            assert "text scores" not in result["reused"], "re-indexed library reused old text scores"
            # Independent recalculation provides a per-file check after same-size replacement.
            index = library.indexes["metadata"]
            fresh = search.text_scores(state.cfg, library.name, index, HINT, state.clap)
            expected = dict(zip(index.paths, fresh))
            for hit in result["results"]:
                assert np.isclose(hit["text_score"], expected[hit["path"]])
                print(f"  {hit['name']}: text={hit['text_score']:.5f}", flush=True)
            print("  PASS: fresh scores match the current files", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("add", "remove", "replace", "all"), default="all")
    args = parser.parse_args()
    samples = sorted(SAMPLES.glob("*.wav"))
    if len(samples) != 5:
        raise SystemExit(f"Expected five sample WAVs in {SAMPLES}; found {len(samples)}")
    # Identify the replacement by filename rather than depending on file order.
    dog = next((i for i, p in enumerate(samples) if p.stem.endswith("_Dog_Bark")), None)
    bell = next((i for i, p in enumerate(samples) if p.stem.endswith("_Church bell")), None)
    if dog is None or bell is None:
        raise SystemExit("Expected the dog bark and church bell sample WAVs")
    no_dog = tuple(i for i in range(len(samples)) if i != dog)
    no_bell = tuple(i for i in range(len(samples)) if i != bell)
    for name, before, after in [
        ("Add", (0, 1, 2, 3), (0, 1, 2, 3, 4)),
        ("Remove", (0, 1, 2, 3, 4), (0, 1, 2, 3)),
        ("Replace", no_bell, no_dog),
    ]:
        if args.scenario in ("all", name.lower()):
            run_scenario(name, before, after, samples)


if __name__ == "__main__":
    main()
