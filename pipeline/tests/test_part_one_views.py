"""The warehouse views that answer each A2 Part One wish-list schema with its own column names."""

import csv

from conftest import tone, write

from vocalocate_pipeline import warehouse
from vocalocate_pipeline.audio import sha256_file
from vocalocate_pipeline.flows import run_dataset_flow, run_library_flow

PART_ONE = {
    "v_sfx_folder": ["folder_id", "folder_path", "index_status", "last_indexed_at"],
    "v_sfx_file": ["folder_id", "sound_id", "file_path", "audio_format", "sample_rate", "bit_depth", "channel",
                   "duration_sec", "category", "description", "tags"],
    "v_vocal_imitation": ["imitation_id", "imitation_path", "duration_sec", "noise_level_db"],
    "v_retrieval_query": ["query_id", "imitation_id", "text_descriptor", "target_sound_id", "split"],
    "v_qvim_benchmark": ["benchmark_query_id", "imitation_id", "target_sound_id", "benchmark_split"],
    "v_imitator_profile": ["imitator_id", "role", "years_experience", "first_language", "music_training",
                           "consent_scope"],
}


def _columns(cfg, view):
    rows = warehouse.query(cfg, "SELECT column_name FROM information_schema.columns "
                                f"WHERE table_name = '{view}' ORDER BY ordinal_position")
    return [r[0] for r in rows]


def test_every_part_one_view_starts_with_the_part_one_columns_in_order(cfg):
    warehouse.load(cfg)  # an empty lake: every view still exists
    for view, cols in PART_ONE.items():
        assert _columns(cfg, view)[: len(cols)] == cols, view
        assert warehouse.query(cfg, f"SELECT count(*) FROM {view}") == [(0,)]
    assert _columns(cfg, "v_retrieval_query") == PART_ONE["v_retrieval_query"] + ["hint_type", "query_type"]
    assert "judged_relevance" not in _columns(cfg, "v_hard_negatives")


def test_sfx_folder_and_file_views_cover_library_and_dataset_sounds(cfg, tmp_path, esc50_dir):
    lib = tmp_path / "MySFX"
    write(lib / "proj" / "door.wav", tone(300, 1, 48_000), 48_000)
    run_library_flow([lib], cfg, library="me", load=False)
    run_dataset_flow("esc50", esc50_dir, cfg)
    folder = warehouse.query(cfg, "SELECT folder_id, folder_path, index_status, library FROM v_sfx_folder")
    assert [r[1:] for r in folder] == [(str(lib.resolve()), "completed", "me")]
    rows = warehouse.query(cfg, "SELECT folder_id, file_path, audio_format, sample_rate, bit_depth, channel, category "
                                "FROM v_sfx_file ORDER BY file_path")
    assert rows[-1] == (folder[0][0], "proj/door.wav", "WAV", 48_000, 16, 1, None)
    assert rows[0][:2] == (None, "audio/1-100-A-0.wav") and rows[0][-1] == "dog"
    assert len(rows) == 5  # four clean ESC-50 clips and the library file


def test_vocal_imitation_view_lists_clean_imitations(cfg, tmp_path):
    team = tmp_path / "team"
    write(team / "a.wav", tone(300, 1, 48_000), 48_000)
    with open(team / "recordings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "imitator_id", "target_group_key", "text_hint", "hint_type"])
        w.writerow(["a.wav", "tanay", "esc50:200", "", ""])
    run_dataset_flow("team", team, cfg)
    rows = warehouse.query(cfg, "SELECT imitation_id, noise_level_db IS NOT NULL, source, imitator_id "
                                "FROM v_vocal_imitation")
    assert rows == [(sha256_file(team / "a.wav"), True, "team", "tanay")]


def test_retrieval_queries_are_recorded_or_generated_from_metadata(cfg, sfx_meta_dir):
    run_dataset_flow("sfx_meta", sfx_meta_dir, cfg)
    rows = warehouse.query(cfg, """
        SELECT q.query_type, q.text_descriptor, q.hint_type, c.source_key, q.split = s.split, length(q.query_id)
        FROM v_retrieval_query q
        JOIN clips c ON c.clip_id = q.target_sound_id
        JOIN splits s ON s.clip_id = q.target_sound_id
        ORDER BY ALL""")
    assert rows == [
        ("metadata", "bell, metallic, low", None, "sfx/gong_A.wav", True, 32),
        ("metadata", "bell, metallic, ringing", None, "sfx/bell_A.wav", True, 32),
        ("recorded", "metallic", "quality", "sfx/bell_A.wav", True, 32),
    ]
    # Generated IDs are stable across warehouse rebuilds.
    ids = warehouse.query(cfg, "SELECT query_id FROM v_retrieval_query ORDER BY 1")
    warehouse.load(cfg)
    assert warehouse.query(cfg, "SELECT query_id FROM v_retrieval_query ORDER BY 1") == ids


def test_team_queries_resolve_targets_in_other_sources_and_keep_their_split(cfg, esc50_dir, tmp_path):
    run_dataset_flow("esc50", esc50_dir, cfg)
    team = tmp_path / "team"
    write(team / "a.wav", tone(300, 1, 48_000), 48_000)
    write(team / "b.wav", tone(320, 1, 48_000), 48_000)
    with open(team / "recordings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "imitator_id", "target_group_key", "text_hint", "hint_type", "split"])
        w.writerow(["a.wav", "tanay", "esc50:200", "loud", "loudness", "val"])
        w.writerow(["b.wav", "tanay", "esc50:400", "", "", "test"])  # voice only: a metadata query
    run_dataset_flow("team", team, cfg)
    rows = warehouse.query(cfg, """
        SELECT q.query_type, q.text_descriptor, c.source_key, q.split
        FROM v_retrieval_query q JOIN clips c ON c.clip_id = q.target_sound_id ORDER BY ALL""")
    assert rows == [("metadata", "cow", "audio/4-400-A-3.wav", "test"),
                    ("recorded", "loud", "audio/2-200-A-1.wav", "val")]


def test_qvim_benchmark_view_names_the_challenge_splits(cfg, qvim_dir, tmp_path):
    run_dataset_flow("qvim_dev", qvim_dir, cfg)
    assert warehouse.query(cfg, "SELECT benchmark_split, count(*) FROM v_qvim_benchmark GROUP BY ALL") == [
        ("Development", 2)]
    # Benchmark pairs are not repeated as generated retrieval queries.
    assert warehouse.query(cfg, "SELECT count(*) FROM v_retrieval_query") == [(0,)]
    root = tmp_path / "EVALDataset" / "EVALCheck2"
    write(root / "Items" / "Birds" / "b.wav", tone(500, 2, 32_000), 32_000)
    write(root / "Queries" / "Birds" / "q.wav", tone(520, 1, 32_000), 32_000)
    with open(root / "EVAL Dataset - EVAL Dataset.csv", "w", newline="", encoding="utf-8") as f:
        f.write(",,,Queries,,\n")
        w = csv.writer(f)
        w.writerow(["Label", "Class", "Items", "Query 1", "Query 2", "Query 3"])
        w.writerow(["1", "Birds", "b.wav", "q.wav", "", ""])
    run_dataset_flow("qvim_eval", root.parent, cfg)
    rows = warehouse.query(cfg, "SELECT benchmark_split, benchmark_query_id = imitation_id, count(*) "
                                "FROM v_qvim_benchmark GROUP BY ALL ORDER BY ALL")
    assert rows == [("Development", True, 2), ("Evaluation", True, 1)]
