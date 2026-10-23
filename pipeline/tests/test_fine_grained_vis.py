from vocalocate_pipeline import warehouse
from vocalocate_pipeline.flows import run_dataset_flow
from vocalocate_pipeline.ingest import ingest
from vocalocate_pipeline.ingest.fine_grained_vis import class_category


def _name(key: str) -> str:
    return key.split("/")[-1]


def test_class_names_match_vimsketch():
    assert class_category("001Animal_Domestic animals_ pets_Cat_Hiss") == "Animal_Domestic animals_pets_Cat_Hiss"
    assert (class_category("009Animal_Livestock_ farm animals_ working animals_Fowl_Chicken_ rooster_Cluck")
            == "Animal_Livestock_farm animals_working animals_Fowl_Chicken_rooster_Cluck")


def test_every_recording_is_a_reference_with_its_own_imitations(fgvis_dir):
    result = ingest("fgvis", fgvis_dir)
    files = {_name(f.source_key): f for f in result.files}
    assert sorted(n for n, f in files.items() if f.role == "reference") == ["caw_a.wav", "caw_b.wav", "hiss_a.wav",
                                                                             "hiss_b.wav"]
    assert files["001Animal_Domestic animals_ pets_Cat_Hiss-3.wav"].role == "unpaired_imitation"
    # Each imitation pairs with its own recording, not with the class.
    assert files["001Animal_Domestic animals_ pets_Cat_Hiss-2.wav"].group_key == files["hiss_b.wav"].group_key
    assert files["hiss_a.wav"].group_key != files["hiss_b.wav"].group_key
    assert files["001Animal_Domestic animals_ pets_Cat_Hiss-1.wav"].imitator_id == "fgvis:7"
    meta = {_name(m["source_key"]): m for m in result.metadata}
    assert meta["hiss_a.wav"]["category"] == meta["hiss_b.wav"]["category"] == "Animal_Domestic animals_pets_Cat_Hiss"
    assert meta["hiss_a.wav"]["tags"] == ["Animal", "Domestic animals", "pets", "Cat", "Hiss"]
    assert meta["caw_a.wav"]["recording_id"] == "freesound:2002"
    assert sorted(q["text_hint"] for q in result.queries) == ["a crow", "air leaking"]


def test_flow_puts_everything_in_train_and_builds_pairs(cfg, fgvis_dir):
    run_dataset_flow("fgvis", fgvis_dir, cfg)
    assert warehouse.query(cfg, "SELECT DISTINCT split FROM splits WHERE source = 'fgvis'") == [("train",)]
    assert warehouse.query(cfg, "SELECT count(*) FROM v_training_pairs WHERE source = 'fgvis'")[0][0] == 4
    # The practice take is a negative linked to the recording it imitates.
    rows = warehouse.query(cfg, "SELECT category FROM v_unpaired_imitations WHERE source = 'fgvis'")
    assert rows == [("Animal_Domestic animals_pets_Cat_Hiss",)]


def test_with_vimsketch_a_class_shares_one_category(cfg, fgvis_dir, vimsketch_release):
    run_dataset_flow("vimsketch", vimsketch_release, cfg)
    run_dataset_flow("fgvis", fgvis_dir, cfg)
    rows = warehouse.query(cfg, """SELECT source, count(*) FROM v_sfx_candidates
                                   WHERE category = 'Animal_Domestic animals_pets_Cat_Hiss' GROUP BY ALL ORDER BY ALL""")
    assert rows == [("fgvis", 2), ("vimsketch", 1)]
    assert warehouse.query(cfg, "SELECT count(*) FROM v_cross_source_duplicates")[0][0] == 0
