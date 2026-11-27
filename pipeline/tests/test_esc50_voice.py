import csv
import shutil

import duckdb
import pytest

from vocalocate_pipeline.flow_functions.splits import split_for_group
from vocalocate_pipeline.flows import run_dataset_flow
from vocalocate_pipeline.ingest import ingest

KEY = "vocal_imitation/{}/{}"


def test_references_are_the_imitated_esc50_clips_grouped_by_freesound_upload(esc50_voice_dir):
    result = ingest("esc50_voice", esc50_voice_dir)
    refs = {f.source_key: f for f in result.files if f.role == "reference"}
    assert sorted(refs) == ["ESC-50/audio/1-100-A-0.wav", "ESC-50/audio/1-100-B-0.wav", "ESC-50/audio/2-200-A-1.wav"]
    assert result.missing == ["ESC-50/audio/5-500-A-4.wav"]
    assert {f.group_key for k, f in refs.items() if "-100-" in k} == {"esc50_voice:freesound:100"}
    assert refs["ESC-50/audio/2-200-A-1.wav"].label == "rooster"
    meta = {m["source_key"]: m for m in result.metadata}
    assert meta["ESC-50/audio/1-100-B-0.wav"]["recording_id"] == "freesound:100"
    assert meta["ESC-50/audio/1-100-B-0.wav"]["category"] == "dog"


def test_each_imitation_targets_its_exact_take_and_hashes_its_split_by_the_vis_group(esc50_voice_dir):
    result = ingest("esc50_voice", esc50_voice_dir)
    imitations = {f.source_key: f for f in result.files if f.role == "imitation"}
    assert len(imitations) == 6
    take_b = KEY.format("001", "1-100-B-0.wav_001.wav")
    assert result.targets[take_b] == "ESC-50/audio/1-100-B-0.wav"
    assert imitations[take_b].imitator_id == "esc50_voice:001"
    assert imitations[take_b].group_key == "esc50_voice:freesound:100"
    assert result.split_groups[take_b] == result.split_groups["ESC-50/audio/1-100-A-0.wav"] == "vis:freesound:100"
    assert result.imitators == [{"imitator_id": "esc50_voice:000", "first_language": "Japanese"},
                                {"imitator_id": "esc50_voice:001", "first_language": "Japanese"}]


def test_esc50_can_sit_next_to_the_release(esc50_voice_dir, tmp_path):
    shutil.move(esc50_voice_dir / "ESC-50", tmp_path / "datasets" / "ESC-50")
    shutil.move(esc50_voice_dir, tmp_path / "datasets" / "ESC-50-Voice")
    result = ingest("esc50_voice", tmp_path / "datasets" / "ESC-50-Voice" / "vocal_imitation")
    assert "../../ESC-50/audio/1-100-A-0.wav" in result.targets.values()


def test_unknown_or_misnamed_imitations_are_errors(esc50_voice_dir):
    bad = esc50_voice_dir / "vocal_imitation" / "000" / "9-999-A-9_000.wav"
    shutil.copy(esc50_voice_dir / "vocal_imitation" / "000" / "1-100-A-0_000.wav", bad)
    with pytest.raises(ValueError, match="not in"):
        ingest("esc50_voice", esc50_voice_dir)
    bad.rename(bad.with_name("1-100-A-0_001.wav"))  # names speaker 001 but sits in folder 000
    with pytest.raises(ValueError, match="_000.wav"):
        ingest("esc50_voice", esc50_voice_dir)


def test_flow_pairs_each_imitation_with_its_take_and_keeps_an_upload_in_one_split(cfg, esc50_voice_dir):
    run_dataset_flow("esc50_voice", esc50_voice_dir, cfg)
    con = duckdb.connect(str(cfg.warehouse_path), read_only=True)
    pairs = con.execute("""
        SELECT i.source_key, r.source_key FROM pairs p
        JOIN clips i ON i.clip_id = p.imitation_clip_id JOIN clips r ON r.clip_id = p.reference_clip_id
        ORDER BY 1""").fetchall()
    splits = con.execute("SELECT DISTINCT group_key, split FROM splits").fetchall()
    con.close()
    assert [(i.split("/")[-1], r.split("/")[-1]) for i, r in pairs] == [
        ("1-100-A-0_000.wav", "1-100-A-0.wav"), ("1-100-B-0_000.wav", "1-100-B-0.wav"),
        ("2-200-A-1_000.wav", "2-200-A-1.wav"), ("1-100-A-0_001.wav", "1-100-A-0.wav"),
        ("1-100-B-0.wav_001.wav", "1-100-B-0.wav"),
    ]  # the imitation of the missing 5-500-A-4 has no pair
    by_group = dict(splits)
    assert len(splits) == len(by_group)  # one split per group
    assert by_group["vis:freesound:100"] == split_for_group("vis:freesound:100", cfg.splits)


def test_an_upload_shared_with_the_vocal_imitation_set_shares_its_split(cfg, esc50_voice_dir, vis_dir):
    licence = vis_dir / "original_recordings_license.csv"
    rows = list(csv.reader(open(licence, newline="", encoding="utf-8")))
    rows[2][-1] = "200"  # other_dog.wav is Freesound upload 200, like ESC-50's 2-200-A-1.wav
    with open(licence, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    run_dataset_flow("vis", vis_dir, cfg, load=False)
    run_dataset_flow("esc50_voice", esc50_voice_dir, cfg)
    con = duckdb.connect(str(cfg.warehouse_path), read_only=True)
    shared = con.execute("SELECT recording_id, sources, n_clips, splits FROM v_shared_recordings").fetchall()
    con.close()
    assert len(shared) == 1
    recording, sources, n_clips, splits = shared[0]
    assert (recording, sources, n_clips, len(splits)) == ("freesound:200", ["esc50_voice", "vis"], 2, 1)
