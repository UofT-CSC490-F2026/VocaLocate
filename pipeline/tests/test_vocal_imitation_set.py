from vocalocate_pipeline import warehouse
from vocalocate_pipeline.flows import run_dataset_flow
from vocalocate_pipeline.ingest import ingest


def test_vis_maps_ontology_freesound_ids_and_descriptions(vis_dir):
    result = ingest("vis", vis_dir)
    roles = {f.source_key.split("/")[-1]: f.role for f in result.files}
    assert roles == {"ref_dog.wav": "reference", "other_dog.wav": "library", "ref_cat.wav": "reference",
                     "other_cat.wav": "library", "im1.wav": "imitation", "im2.wav": "imitation", "im3.wav": "imitation",
                     "im4.wav": "unpaired_imitation"}
    meta = {m["source_key"].split("/")[-1]: m for m in result.metadata}
    assert meta["other_dog.wav"]["category"] == "Bark"
    assert meta["other_dog.wav"]["tags"] == ["Animal", "Domestic animals, pets", "Dog", "Bark"]
    assert meta["ref_cat.wav"]["recording_id"] == "freesound:1002"
    # Participants' own descriptions become text hints; empty ones are skipped.
    assert sorted(q["text_hint"] for q in result.queries) == ["a dog barking", "cat meowing"]


def test_vis_flow_builds_pairs_and_related_candidates(cfg, vis_dir):
    run_dataset_flow("vis", vis_dir, cfg)
    assert warehouse.query(cfg, "SELECT count(*) FROM v_training_pairs WHERE source = 'vis'")[0][0] == 3
    rows = warehouse.query(cfg, "SELECT role, category FROM v_sfx_candidates WHERE source = 'vis' ORDER BY ALL")
    assert rows == [("library", "Bark"), ("library", "Meow"), ("reference", "Bark"), ("reference", "Meow")]
    assert warehouse.query(cfg, "SELECT count(*) FROM queries WHERE source = 'vis'")[0][0] == 2


def test_vis_keeps_rejected_imitations_as_unpaired_negatives(cfg, vis_dir):
    run_dataset_flow("vis", vis_dir, cfg)
    # The rejected take never forms a pair...
    assert warehouse.query(cfg, "SELECT count(*) FROM v_training_pairs WHERE source = 'vis'")[0][0] == 3
    # ...but is available as a negative, linked to the sound it imitates and in that sound's split.
    rows = warehouse.query(cfg, """SELECT u.category, u.split = s.split FROM v_unpaired_imitations u
                                   JOIN splits s ON s.clip_id = u.target_clip_id""")
    assert rows == [("Meow", True)]


def test_vis_participant_survey_fills_imitator_profiles(vis_dir):
    result = ingest("vis", vis_dir)
    profiles = {r["imitator_id"]: r for r in result.imitators}
    # p2 took no survey and the listener recorded no imitation in this set.
    assert set(profiles) == {"p1", "p3"}
    assert profiles["p1"] == {"imitator_id": "p1", "role": "crowd worker (Amazon Mechanical Turk)",
                              "years_experience": 2, "first_language": None, "music_training": False,
                              "consent_scope": None}
    # 72 "years" of music technology at age 30 is not believable: left empty.
    assert profiles["p3"]["years_experience"] is None and profiles["p3"]["music_training"] is True


def test_rounded_participant_ids_are_matched_to_the_survey():
    from vocalocate_pipeline.ingest.vocal_imitation_set import rounded_id_matcher

    # vocal_imitations.txt stores IDs as floats rounded to six significant digits.
    ids = {"5734240000000000.0", "4875300000000000.0"}
    match = rounded_id_matcher(ids, ["5734238123456789", "4875301000000000", "4875302000000000", "1"])
    assert match("5734238123456789") == "5734240000000000.0"
    assert match("4875301000000000") is None  # two survey IDs round to the same value: ambiguous
    assert match("1") is None


def test_vis_flow_writes_imitators(cfg, vis_dir):
    run_dataset_flow("vis", vis_dir, cfg)
    rows = warehouse.query(cfg, "SELECT imitator_id, source, years_experience FROM imitators ORDER BY 1")
    assert rows == [("p1", "vis", 2), ("p3", "vis", None)]
