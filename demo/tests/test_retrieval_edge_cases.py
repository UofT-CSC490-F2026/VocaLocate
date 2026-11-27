import csv
from unittest.mock import Mock

import duckdb
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vocalocate_demo import demo_flow, experiment, synth
from vocalocate_demo.data import Dataset, load_dataset, rebase_lake_path
from vocalocate_demo.demo_flow import demo_searches
from vocalocate_demo.evaluate import rank_metrics, score_metrics
from vocalocate_demo.mining import MiningConfig
from vocalocate_demo.search import embed_query
from vocalocate_demo.text import ClapScorer
from vocalocate_demo.train import TrainConfig
from vocalocate_pipeline import warehouse
from vocalocate_pipeline.config import PipelineConfig


def test_existing_warehouse_without_unpaired_view_can_still_load_pairs(tmp_path):
    cfg = PipelineConfig(lake_root=tmp_path / "lake")
    cfg.warehouse_path.parent.mkdir(parents=True)
    with duckdb.connect(str(cfg.warehouse_path)) as con:
        con.execute("""
            CREATE TABLE v_training_pairs AS SELECT
                'vis' AS source, 'voice' AS imitation_clip_id, 'sound' AS reference_clip_id,
                'train' AS split, 'features/voice.npy' AS imitation_feature_path,
                'features/sound.npy' AS reference_feature_path;
            CREATE TABLE v_sfx_candidates AS SELECT
                'vis' AS source, 'sound' AS clip_id, 'reference' AS role, 'features/sound.npy' AS feature_path;
            CREATE TABLE queries AS SELECT 'vis' AS source, 'voice' AS imitation_clip_id, 'bark' AS text_hint;
        """)
    dataset = load_dataset(cfg, ["vis"])
    assert dataset.split_pairs("train").imitation_clip_id.tolist() == ["voice"]
    assert dataset.split_unpaired("train").empty
    assert dataset.hints == {"voice": "bark"}
    assert rebase_lake_path("features/voice.npy", cfg.lake_root) == "features/voice.npy"


def test_dataset_loading_reports_an_empty_source(tmp_path):
    cfg = PipelineConfig(lake_root=tmp_path / "lake")
    warehouse.load(cfg)
    with pytest.raises(ValueError, match=r"No training pairs for \['vis'\]"):
        load_dataset(cfg, ["vis"])


def test_metrics_report_empty_query_set():
    targets = np.empty(0, dtype=int)
    assert rank_metrics(np.empty((0, 2)), np.eye(2), targets, []) == {"n_queries": 0}
    assert score_metrics(np.empty((0, 2)), targets, []) == {"n_queries": 0}


def test_search_rejects_silent_query_before_embedding(tmp_path):
    cfg = PipelineConfig(lake_root=tmp_path / "lake")
    query = tmp_path / "silent.wav"
    sf.write(query, np.zeros(32_000), 32_000)
    with pytest.raises(ValueError, match="silent.wav rejected by cleaning: silent"):
        embed_query(query, None, TrainConfig(), cfg)


def test_recipe_text_requires_at_least_one_requested_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "load_dataset", lambda cfg, sources: Dataset(pd.DataFrame(), pd.DataFrame()))
    cfg = PipelineConfig(lake_root=tmp_path / "lake")
    with pytest.raises(FileNotFoundError, match="no recipe checkpoints"):
        experiment.run_recipe_text_branch(cfg, "vis", [tmp_path], ClapScorer(backend="fake"), names=["greif"])


def test_text_branch_skips_splits_without_hints(tmp_path, caplog):
    cfg = PipelineConfig(lake_root=tmp_path / "lake")
    clip = tmp_path / "sound.wav"
    signal = 0.5 * np.sin(2 * np.pi * 400 * np.arange(16_000) / 16_000)
    sf.write(clip, signal, 16_000)
    dataset = Dataset(
        pairs=pd.DataFrame([
            {"imitation_clip_id": f"{split}-voice", "reference_clip_id": f"{split}-sound", "split": split,
             "imitation_feature_path": None, "reference_feature_path": None}
            for split in ("val", "test")
        ]),
        candidates=pd.DataFrame([
            {"clip_id": f"{split}-sound", "clean_path": str(clip), "feature_path": None, "split": split,
             "tags": ["metal"], "category": "bell", "recording_id": f"{split}-recording"}
            for split in ("val", "test")
        ]),
        hints={"val-voice": "bell"},
    )
    clap = ClapScorer(backend="fake")
    result = experiment.run_text_branch(cfg, "vis", dataset, {}, TrainConfig(device="cpu"), clap)

    assert result == {"clap": clap.version, "n_queries": {"val": 1, "test": 0}, "models": {}}
    assert "no val or test imitations with hints" in caplog.text


def test_demo_and_benchmark_handle_missing_models(tmp_path, monkeypatch):
    cfg = PipelineConfig(lake_root=tmp_path / "lake")
    assert demo_searches(cfg, tmp_path, []) == {"library": {}, "queries": []}
    monkeypatch.setattr(experiment, "load_dataset", lambda cfg, sources: Dataset(pd.DataFrame(), pd.DataFrame()))

    (cfg.lake_root / "models" / "vis").mkdir(parents=True)
    result = experiment.run_benchmark(cfg, "vis", names=("baseline",), device="cpu")
    assert result["models"] == {}
    assert (cfg.lake_root / "models/vis/benchmark_qvim_dev.json").is_file()


def test_sweep_grid_keeps_controls_fixed_and_covers_all_band_cutoffs():
    base = MiningConfig(n_neg=4, seed=51)
    options = experiment.sweep_grid(base)
    assert {(cfg.band, cfg.dup_quantile) for cfg in options} == {
        (band, cutoff) for band in experiment.SWEEP_BANDS for cutoff in experiment.SWEEP_DUP_QUANTILES
    }
    assert all(cfg.n_neg == 4 and cfg.seed == 51 for cfg in options)


def test_synthetic_renderer_rejects_unknown_category_and_variant():
    rng = np.random.default_rng(490)
    with pytest.raises(ValueError, match="whale"):
        synth.render_sfx("whale", synth._rng_params("whale", rng), rng)
    with pytest.raises(ValueError, match="stretched"):
        synth.make_variant(np.ones(100), "stretched", rng)


def test_messy_library_queries_skip_targets_outside_the_library(tmp_path, monkeypatch):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    candidates = [("bell", str(clip), "bells"), ("dog", str(clip), "dogs")]
    pairs = [("im-bell", str(clip), "bell", "bells", "ding"), ("im-dog", str(clip), "dog", "dogs", None),
             ("im-gone", str(clip), "held-back", "cars", "vroom")]  # its reference is not a test-split SFX
    monkeypatch.setattr(warehouse, "query", Mock(side_effect=[candidates, pairs]))

    # Ask for more queries than there are usable references: all usable ones are picked, no more.
    queries = demo_flow.build_messy_library(PipelineConfig(lake_root=tmp_path / "lake"), tmp_path / "run", n_queries=5)

    assert sorted(q["category"] for q in queries) == ["bells", "dogs"]
    library = {str(p.resolve()) for p in (tmp_path / "run" / "library").rglob("*.wav")}
    assert {q["target"] for q in queries} == library and len(library) == 2
    with open(tmp_path / "run" / "queries" / "queries.csv", newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 2


def test_demo_reuses_an_existing_dataset_and_reports_voice_only_ranks(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / demo_flow.SOURCE
    data_dir.mkdir(parents=True)
    (data_dir / "metadata.csv").write_text("already generated", encoding="utf-8")
    generate, flow = Mock(), Mock()
    flow.return_value.as_dict.return_value = {}
    search_demo = {"library": {"baseline": {"files": 3, "embedded": 3, "sec": 0.1}},
                   "queries": [{"query": str(tmp_path / "q.wav"), "category": "bells", "hint": "ding",
                                "results": {"baseline": {"target_rank": 2, "top": []}}}]}
    monkeypatch.setattr(demo_flow, "generate", generate)
    monkeypatch.setattr(demo_flow, "run_dataset_flow", flow)
    monkeypatch.setattr(demo_flow, "run_experiment", Mock(return_value={
        "source": demo_flow.SOURCE, "device": "cpu", "data": {}, "models": {}, "mining": {}}))
    monkeypatch.setattr(warehouse, "query", Mock(return_value=[]))
    monkeypatch.setattr(demo_flow, "build_messy_library", Mock(return_value=search_demo["queries"]))
    monkeypatch.setattr(demo_flow, "demo_searches", Mock(return_value=search_demo))
    cfg = PipelineConfig(lake_root=tmp_path / "lake")

    demo_flow.run_demo(tmp_path, cfg, TrainConfig(device="cpu"), MiningConfig())

    generate.assert_not_called()
    flow.assert_called_once_with(demo_flow.SOURCE, data_dir, cfg)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| `q.wav` | bells | 2 |" in report  # no text rank, so no hint and no "a → b" legend
    assert "rank with the imitation alone" not in report


def test_sweep_keeps_the_first_grid_point_with_the_best_validation_mrr(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from vocalocate_demo import train

    features = str(tmp_path / "clip.npy")
    dataset = Dataset(
        pairs=pd.DataFrame([{"source": "vis", "split": "train", "imitation_clip_id": "voice", "reference_clip_id": "bell",
                             "imitation_feature_path": features, "reference_feature_path": features}]),
        candidates=pd.DataFrame([{"clip_id": "bell", "split": "train", "category": "bells", "recording_id": "r",
                                  "tags": ["metal"], "feature_path": features}]),
    )
    monkeypatch.setattr(experiment, "load_dataset", lambda cfg, sources: dataset)
    monkeypatch.setattr(train, "train_baseline", Mock(return_value=("baseline", [])))
    monkeypatch.setattr(train, "embed", Mock(return_value=np.zeros((1, 2))))
    save = Mock()
    monkeypatch.setattr(train, "save", save)
    monkeypatch.setattr(experiment, "mine", Mock(return_value=SimpleNamespace(stats={}, negatives=[], rows=[])))
    val_mrrs = [0.3, 0.8, 0.5, 0.8, 0.1, 0.2]  # grid points 1 and 3 tie for best
    monkeypatch.setattr(train, "finetune", Mock(side_effect=[(f"model{i}", [{"val_mrr": v}])
                                                             for i, v in enumerate(val_mrrs)]))
    monkeypatch.setattr(experiment, "_store", Mock())
    monkeypatch.setattr(experiment, "_score", Mock(return_value={}))
    monkeypatch.setattr(warehouse, "load", Mock())
    cfg, mcfg = PipelineConfig(lake_root=tmp_path / "lake"), MiningConfig()
    (cfg.lake_root / "models" / "vis").mkdir(parents=True)  # T.save, mocked here, would create it

    results = experiment.run_experiment(cfg, "vis", TrainConfig(device="cpu"), mcfg, strategies=("metadata",),
                                        sweep=True, clap=None)

    grid = experiment.sweep_grid(mcfg)
    assert [row["val_mrr"] for row in results["sweep"]] == val_mrrs
    model, path, _, extra = save.call_args.args
    assert (model, path.name, extra["mining"]["band"]) == ("model1", "metadata.pt", grid[1].band)
    assert extra["mining"]["dup_quantile"] == grid[1].dup_quantile
    assert "text" not in results and not (tmp_path / "lake" / "models" / "vis" / "fusion.json").exists()


def test_benchmark_without_published_scores_names_its_file_after_the_source(tmp_path, monkeypatch):
    cfg = PipelineConfig(lake_root=tmp_path / "lake")
    monkeypatch.setattr(experiment, "load_dataset", lambda cfg, sources: Dataset(pd.DataFrame(), pd.DataFrame()))
    (cfg.lake_root / "models" / "vis").mkdir(parents=True)

    result = experiment.run_benchmark(cfg, "vis", bench_source="vimsketch", names=("baseline",), device="cpu")

    assert result["benchmark"] == "vimsketch" and "published_mrr" not in result
    assert (cfg.lake_root / "models/vis/benchmark_vimsketch.json").is_file()
