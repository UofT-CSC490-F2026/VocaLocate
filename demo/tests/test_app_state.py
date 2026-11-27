"""AppState and the server's edges without a browser: boot failures, CLAP loading, settings validation, model
import, re-index races, library and model defaults, the native helpers (folder picker, reveal) and the
cached-index lookups in search.py. Several of these pin fixes from the code review of the desktop app.
"""

import json
import queue
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vocalocate_pipeline.config import PipelineConfig
from vocalocate_pipeline.lake import Lake
from vocalocate_demo import app as app_mod
from vocalocate_demo import desktop, search
from vocalocate_demo import train as T
from vocalocate_demo.app import AppState, Library
from vocalocate_demo.model import DualEncoder


def _checkpoints(folder: Path, versions: dict[str, str]) -> Path:
    tcfg = T.TrainConfig(frames=64, device="cpu")
    for name, version in versions.items():
        extra = {"model_version": version} if version else {}
        T.save(DualEncoder(dim=16, width=4, frames=64), folder / f"{name}.pt", tcfg, extra)
    return folder


def _state(home: Path, **kw) -> AppState:
    kw.setdefault("text_auto", False)
    return AppState(PipelineConfig(lake_root=home / "lake", workers=1), desktop.SOURCE, home=home,
                    autostart_dir=home / "fake_startup", **kw)


def _sounds(root: Path, n: int = 6) -> Path:
    rng = np.random.default_rng(1)
    root.mkdir(parents=True)
    for i in range(n):
        t = np.arange(16_000) / 32_000
        wave = 0.3 * np.sin(2 * np.pi * (220 + 70 * i) * t) + 0.01 * rng.standard_normal(t.size)
        sf.write(root / f"s{i}.wav", wave.astype(np.float32), 32_000)
    return root


def _wait(cond, timeout=180.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return False


# -- boot -----------------------------------------------------------------------------------------------
def test_boot_failure_is_reported_and_restored_libraries_wait_for_models(tmp_path, monkeypatch):
    desktop.save_settings(tmp_path, {**desktop.DEFAULT_SETTINGS, "libraries": [{"name": "sfx", "roots": [str(tmp_path)]}]})
    state = _state(tmp_path)

    def broken():
        raise ImportError("torch is missing")

    monkeypatch.setattr(app_mod, "load_engine", broken)
    state._boot()
    assert state.boot == {"state": "ready", "message": "Could not load the models: ImportError: torch is missing"}
    assert state.libraries["sfx"].message.startswith("Waiting for models")


def test_attach_cached_skips_known_models_and_survives_a_broken_cache(tmp_path, monkeypatch):
    state = _state(tmp_path)
    lib = Library(name="sfx", roots=[str(tmp_path)])
    lib.indexes["metadata"] = "kept"
    monkeypatch.setattr(app_mod, "cached_index", lambda *a, **k: pytest.fail("looked up a model it already has"))
    state._attach_cached(lib, "metadata", "v1")

    def broken(*a, **k):
        raise OSError("corrupt part")

    monkeypatch.setattr(app_mod, "cached_index", broken)
    state._attach_cached(lib, "retrieval", "v1")
    assert lib.indexes == {"metadata": "kept"}


# -- CLAP -----------------------------------------------------------------------------------------------
def test_clap_load_starts_once(tmp_path):
    state = _state(tmp_path, text_auto=True)
    started, gate = [], threading.Event()
    state._load_clap = lambda: (started.append(1), gate.wait(10))
    state._maybe_load_clap()
    state._maybe_load_clap()  # already loading
    state.update_settings({"text": True})  # a toggle while it loads
    assert state.text_state == "loading"
    assert _wait(lambda: started == [1], 5) and started == [1]
    gate.set()
    state.clap = object()
    state.text_state = "off"
    state._start_clap(("off",))  # CLAP already there
    assert started == [1]


def test_load_clap_ready_missing_failed_and_loaded(tmp_path, monkeypatch):
    import importlib.util

    from vocalocate_demo import text

    state = _state(tmp_path)
    state.clap = "already"
    state._load_clap()
    assert state.text_state == "ready"

    state.clap = None
    real_find = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "transformers" else real_find(name, *a))
    state._load_clap()
    assert state.text_state == "off" and "transformers" in state.text_error

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object())

    def no_weights(**kw):
        raise OSError("no internet")

    monkeypatch.setattr(text, "ClapScorer", no_weights)
    state._load_clap()
    assert state.text_state == "error" and state.text_error == "OSError: no internet"

    made = []
    monkeypatch.setattr(text, "ClapScorer", lambda **kw: made.append(kw) or "clap")
    state._load_clap()
    assert state.clap == "clap" and state.text_state == "ready" and state.text_error == ""
    assert made == [{"device": state.device}]


# -- settings -------------------------------------------------------------------------------------------
def test_rejected_settings_change_nothing(tmp_path):
    state = AppState(PipelineConfig(lake_root=tmp_path / "lake"), "sfx_meta")  # no data dir: no autostart
    before = dict(state.settings)
    with pytest.raises(ValueError, match="only available in the Windows desktop app"):
        state.update_settings({"model": "baseline", "autostart": True})
    with pytest.raises(ValueError, match="device must be"):
        state.update_settings({"model": "baseline", "device": "turbo"})
    assert state.settings == before


def test_text_setting_off_saves_and_starts_nothing(tmp_path):
    state = _state(tmp_path)
    state._load_clap = lambda: pytest.fail("loaded CLAP with hints off")
    assert state.update_settings({"text": False})["text"] is False
    assert desktop.load_settings(tmp_path)["text"] is False


def test_concurrent_library_changes_all_reach_settings_json(tmp_path, monkeypatch):
    state = _state(tmp_path)
    monkeypatch.setattr(state, "_queue_index", lambda lib: None)
    folders = []
    for i in range(8):
        (tmp_path / f"f{i}").mkdir()
        folders.append(str(tmp_path / f"f{i}"))
    threads = [threading.Thread(target=state.add_library, args=(f"lib{i}", [f])) for i, f in enumerate(folders)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(e["name"] for e in desktop.load_settings(tmp_path)["libraries"]) == [f"lib{i}" for i in range(8)]


def test_bad_fusion_json_gives_no_alphas(tmp_path):
    folder = tmp_path / "lake" / "models" / desktop.SOURCE
    folder.mkdir(parents=True)
    (folder / "fusion.json").write_text("{not json", encoding="utf-8")
    assert _state(tmp_path).alphas == {}


# -- model import ---------------------------------------------------------------------------------------
def test_import_refuses_an_explicit_models_folder(tmp_path):
    run = _checkpoints(tmp_path / "recipe-492", {"shared": "v-shared"})
    state = AppState(PipelineConfig(lake_root=tmp_path / "lake"), "sfx_meta", models=run)
    with pytest.raises(ValueError, match="does not change them"):
        state.import_models(str(_checkpoints(tmp_path / "other", {"shared": "v-other"})))
    assert T.load(run / "shared.pt", device="cpu")[2]["model_version"] == "v-shared"


def test_import_rejects_a_missing_folder(tmp_path):
    with pytest.raises(ValueError, match="not a folder"):
        _state(tmp_path).import_models(str(tmp_path / "nope"))


@pytest.mark.parametrize("bad", ["corrupt", "no_version"])
def test_failed_import_leaves_installed_models_untouched(tmp_path, bad):
    home = tmp_path / "home"
    state = _state(home)
    state.import_models(str(_checkpoints(tmp_path / "v1", {"metadata": "m1", "retrieval": "r1"})))
    new = _checkpoints(tmp_path / "v2", {"metadata": "m2", "retrieval": "r2" if bad == "corrupt" else None})
    if bad == "corrupt":
        (new / "retrieval.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="retrieval.pt is not a VocaLocate model checkpoint"):
        state.import_models(str(new))
    installed = state.models_dir
    assert sorted(p.name for p in installed.iterdir()) == ["metadata.pt", "retrieval.pt"]  # no .importing left
    assert T.load(installed / "metadata.pt", device="cpu")[2]["model_version"] == "m1"  # not half-imported
    assert state.model("metadata")[2] == "m1"


def test_reimport_with_new_versions_drops_old_indexes_and_rescans(tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = _state(home)
    queued = []
    monkeypatch.setattr(state, "_queue_index", queued.append)
    state.import_models(str(_checkpoints(tmp_path / "v1", {"metadata": "m1"})))
    (tmp_path / "sfx").mkdir()
    lib = Library(name="sfx", roots=[str((tmp_path / "sfx").resolve())])
    lib.indexes["metadata"] = "old index"
    gone = Library(name="unplugged", roots=[str(tmp_path / "unplugged")])
    state.libraries.update(sfx=lib, unplugged=gone)

    state.import_models(str(_checkpoints(tmp_path / "v2", {"metadata": "m2"})))
    assert lib.indexes == {}  # built with the old checkpoint
    assert [q.name for q in queued] == ["sfx"] and state.libraries["sfx"] is queued[0]
    assert gone.state == "error" and "not a folder" in gone.message


def test_reimporting_the_installed_folder_reloads_in_place(tmp_path, monkeypatch):
    state = _state(tmp_path / "home")
    monkeypatch.setattr(state, "_queue_index", lambda lib: None)
    state.import_models(str(_checkpoints(tmp_path / "v1", {"metadata": "m1"})))
    res = state.import_models(str(state.models_dir))
    assert [m["name"] for m in res["models"]] == ["metadata"] and state.model("metadata")[2] == "m1"


# -- libraries and indexing -------------------------------------------------------------------------------
def test_rescan_and_index_skip_unknown_or_replaced_libraries(tmp_path, monkeypatch):
    state = _state(tmp_path)
    with pytest.raises(KeyError, match="unknown library"):
        state.rescan("nope")
    stray = Library(name="stray", roots=[str(tmp_path)])
    state._index(stray)  # never registered (deleted while it waited)
    assert stray.state == "queued" and stray.finished is None


def test_index_stops_when_the_library_is_replaced_mid_way(tmp_path, monkeypatch):
    state = _state(tmp_path / "home")
    monkeypatch.setattr(state, "_queue_index", lambda lib: None)
    state.import_models(str(_checkpoints(tmp_path / "v1", {"metadata": "m1"})))
    lib = state.add_library("sfx", [str(_sounds(tmp_path / "sfx"))])

    def replaced(*a, **k):
        state.libraries["sfx"] = Library(name="sfx", roots=lib.roots)
        return types.SimpleNamespace(n_embedded=0, clip_ids=[])

    monkeypatch.setattr(app_mod, "index_library", replaced)
    state._index(lib)
    assert lib.indexes == {} and lib.state != "ready"


def test_default_library_and_model(tmp_path):
    state = _state(tmp_path)
    assert state.default_library() == "default"  # nothing yet: the search reports it
    state.libraries["sfx"] = Library(name="sfx", roots=[])
    assert state.default_library() == "sfx"
    state.libraries["foley"] = Library(name="foley", roots=[])
    with pytest.raises(ValueError, match="choose a library to search: foley, sfx"):
        state.default_library()
    lib = state.libraries["sfx"]
    assert state.resolve_model(lib, "retrieval") == "retrieval"
    assert state.resolve_model(lib, None) == "shared"  # nothing indexed: the saved choice
    state.settings["model"] = None
    assert state.resolve_model(lib, None) == "metadata"


# -- text hints during a re-index ---------------------------------------------------------------------------
def test_text_scores_tolerate_clips_the_new_scan_dropped(monkeypatch):
    from vocalocate_demo import text

    monkeypatch.setattr(search.warehouse, "query", lambda cfg, sql: [("a", "a.wav"), ("c", "c.wav")])
    monkeypatch.setattr(text, "cached_audio_embeddings",
                        lambda cfg, part, ids, paths, clap: np.array([[0.2, 0.0], [0.9, 0.0]][: len(ids)], np.float32))
    clap = types.SimpleNamespace(embed_text=lambda t: np.array([[1.0, 0.0]], np.float32))
    idx = search.LibraryIndex(clip_ids=["a", "b", "c"], paths=["a", "b", "c"], projects=[None] * 3,
                              vectors=np.zeros((3, 2), np.float32), n_embedded=0)
    scores = search.text_scores(None, "sfx", idx, "boom", clap)
    assert np.allclose(scores, [0.2, 0.2, 0.9])  # b (deleted) ranks last, and the scores stay finite
    monkeypatch.setattr(search.warehouse, "query", lambda cfg, sql: [])
    assert np.allclose(search.text_scores(None, "sfx", idx, "boom", clap), 0.0)


def _idx(ids):
    n = len(ids)
    return search.LibraryIndex(clip_ids=list(ids), paths=list(ids), projects=[None] * n,
                               vectors=np.zeros((n, 2), np.float32), n_embedded=0)


def test_hint_clip_vectors_are_computed_once_and_shared(tmp_path, monkeypatch):
    calls, release = [], threading.Event()

    def slow_pass(cfg, library, idx, clap, lock=None, progress=None):
        calls.append(library)
        release.wait(5)
        return np.arange(len(idx.clip_ids)), np.ones((len(idx.clip_ids), 2), np.float32)

    monkeypatch.setattr(app_mod, "clap_audio_for_index", slow_pass)
    state = _state(tmp_path)
    state.clap = types.SimpleNamespace(version="clap:fake")
    state.libraries["lib"] = Library(name="lib", roots=[], state="ready")
    idx = _idx(["a", "b", "c"])
    for _ in range(3):  # searches and the end of indexing all ask; one background pass is queued
        state._ensure_hints("lib", idx)
    got = []
    waiter = threading.Thread(target=lambda: got.append(state._hint_audio("lib", idx)))
    waiter.start()
    assert _wait(lambda: len(calls) == 1, 5)
    assert "lib" in state.hint_prep and state._hint_entry("lib", idx) is None  # searches answer by voice meanwhile
    release.set()
    waiter.join(5)
    assert calls == ["lib"] and got and got[0][1].shape == (3, 2)  # the waiter shared the pass
    assert _wait(lambda: "lib" not in state.hint_prep, 5)
    state._ensure_hints("lib", idx)
    assert state._hint_audio("lib", idx)[1] is got[0][1]
    assert calls == ["lib"]  # later hints reuse the vectors in memory
    state._hint_audio("lib", _idx(["a", "c"]))  # the library changed: vectors are rebuilt for the new index
    assert calls == ["lib", "lib"]
    state.delete_library("lib")
    assert "lib" not in state.hint_vectors


def test_clap_cache_is_only_rewritten_when_it_changes(tmp_path, monkeypatch):
    from vocalocate_demo import text

    paths = []
    for i in range(3):
        p = tmp_path / f"c{i}.wav"
        sf.write(p, np.zeros(4_800, np.float32), 48_000)
        paths.append(str(p))
    cfg = PipelineConfig(lake_root=tmp_path / "lake", workers=1)
    clap = text.ClapScorer(backend="fake")
    writes = []
    real_write = Lake.write
    monkeypatch.setattr(Lake, "write", lambda self, *a, **k: (writes.append(a[1]), real_write(self, *a, **k))[1])
    seen = []
    first = text.cached_audio_embeddings(cfg, "lib__clap", ["a", "b", "c"], paths, clap,
                                         progress=lambda d, t: seen.append((d, t)))
    assert len(writes) == 1 and seen[-1] == (3, 3)
    again = text.cached_audio_embeddings(cfg, "lib__clap", ["a", "b", "c"], paths, clap)
    assert len(writes) == 1 and np.allclose(first, again)  # nothing new: no rewrite
    text.cached_audio_embeddings(cfg, "lib__clap", ["a", "c"], [paths[0], paths[2]], clap)
    assert len(writes) == 2  # a clip left the library: the part is rewritten without it


# -- cached indexes (search.py) -----------------------------------------------------------------------------
def test_cached_index_needs_a_scan_and_a_complete_cache(tmp_path, monkeypatch):
    cfg = PipelineConfig(lake_root=tmp_path / "lake", workers=1)
    assert search._library_rows(cfg, "sfx") == []  # no warehouse yet
    assert search.cached_index(cfg, "sfx", "v1") is None

    state = _state(tmp_path)
    state.import_models(str(_checkpoints(tmp_path / "v1", {"metadata": "m1"})))
    state.add_library("sfx", [str(_sounds(tmp_path / "sfx"))])
    assert _wait(lambda: state.libraries["sfx"].state in ("ready", "error"))
    assert state.libraries["sfx"].state == "ready", state.libraries["sfx"].message
    cfg = state.cfg
    full = search.cached_index(cfg, "sfx", "m1")
    assert full is not None and len(full.clip_ids) == 6
    assert search.cached_index(cfg, "sfx", "never-embedded") is None
    lake = Lake(cfg.lake_root)
    part = lake.read("embeddings", "library_sfx__m1")
    lake.write("embeddings", "library_sfx__partial", part.slice(0, 3))
    assert search.cached_index(cfg, "sfx", "partial") is None  # ids would not line up with the other models


# -- native helpers -----------------------------------------------------------------------------------------
def _fake_tk(monkeypatch, askdirectory, tk_fails=False):
    calls = []

    class Root:
        def withdraw(self):
            calls.append("withdraw")

        def attributes(self, *a):
            calls.append(("attributes", a))

        def lift(self):
            pass

        def update(self):
            pass

    def make_root():
        if tk_fails:
            raise RuntimeError("no display")
        return Root()

    filedialog = types.SimpleNamespace(askdirectory=askdirectory)
    tkinter = types.SimpleNamespace(Tk=make_root, filedialog=filedialog)
    monkeypatch.setitem(sys.modules, "tkinter", tkinter)
    monkeypatch.setitem(sys.modules, "tkinter.filedialog", filedialog)
    monkeypatch.setattr(sys, "platform", "win32")
    return calls


def test_folder_picker_answers_cancels_and_survives_errors(monkeypatch, tmp_path):
    answers = queue.Queue()
    for a in (str(tmp_path), "", RuntimeError("dialog crashed")):
        answers.put(a)

    def askdirectory(**kw):
        a = answers.get()
        if isinstance(a, Exception):
            raise a
        return a

    calls = _fake_tk(monkeypatch, askdirectory)
    picker = app_mod._FolderPicker()
    assert picker("Sounds") == str(tmp_path)
    thread = picker._thread
    assert picker("Sounds") is None  # cancelled
    assert picker("Sounds") is None  # the dialog failed; the thread keeps serving
    assert picker._thread is thread and calls[0] == "withdraw"


def test_folder_picker_without_tk_and_with_a_stuck_dialog(monkeypatch):
    _fake_tk(monkeypatch, lambda **kw: pytest.fail("no root, no dialog"), tk_fails=True)
    assert app_mod._FolderPicker()("x") is None

    release = threading.Event()
    _fake_tk(monkeypatch, lambda **kw: (release.wait(10), "")[1])
    picker = app_mod._FolderPicker()
    picker.timeout = 0.2
    try:
        assert picker("first") is None  # the dialog is still open: the request gives up instead of hanging
        assert picker("second") is None  # and does not block the next one behind a lock
    finally:
        release.set()


def test_folder_picker_on_macos_uses_applescript(monkeypatch):
    runs = []
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(app_mod.subprocess, "run",
                        lambda cmd, **kw: runs.append(cmd) or types.SimpleNamespace(returncode=0, stdout="/Users/me/SFX/\n"))
    assert app_mod._FolderPicker()('Pick "sounds"') == "/Users/me/SFX/"
    assert runs[0][:2] == ["osascript", "-e"] and '"Pick sounds"' in runs[0][2]
    monkeypatch.setattr(app_mod.subprocess, "run",
                        lambda cmd, **kw: types.SimpleNamespace(returncode=1, stdout="/stale\n"))
    assert app_mod._pick_folder_macos("x") is None  # cancelled: never the leftover output

    def missing(cmd, **kw):
        raise FileNotFoundError("osascript")

    monkeypatch.setattr(app_mod.subprocess, "run", missing)
    assert app_mod._pick_folder_macos("x") is None


@pytest.mark.parametrize("platform_name, expected", [
    ("win32", 'explorer /select,"{path}"'),
    ("darwin", ["open", "-R", "{path}"]),
    ("linux", ["xdg-open", "{parent}"]),
])
def test_reveal_on_each_platform(monkeypatch, tmp_path, platform_name, expected):
    path = tmp_path / "clip.wav"
    seen = []
    monkeypatch.setattr(sys, "platform", platform_name)
    monkeypatch.setattr(app_mod.subprocess, "Popen", lambda cmd: seen.append(cmd))
    app_mod._reveal(path)
    fill = {"path": str(path), "parent": str(tmp_path)}
    want = expected.format(**fill) if isinstance(expected, str) else [e.format(**fill) for e in expected]
    assert seen == [want]


# -- HTTP edges -----------------------------------------------------------------------------------------------
def test_http_edges(tmp_path, monkeypatch):
    import urllib.error
    import urllib.request

    server = app_mod.serve(_state(tmp_path), port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path, body, method="POST"):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        assert call("/api/settings", ["not", "an", "object"]) == (400, {"error": "send a JSON object"})
        assert call("/api/models/import", {"path": "  "})[0] == 400
        assert call("/api/nothing-here", {}, method="DELETE") == (404, {"error": "not found"})
    finally:
        server.shutdown()
        server.server_close()


def test_serve_free_gives_up_after_the_tried_ports(monkeypatch, tmp_path):
    def busy(state, host, port):
        raise OSError("in use")

    monkeypatch.setattr(app_mod, "serve", busy)
    with pytest.raises(OSError, match="no free port in 9000-9002"):
        app_mod.serve_free(_state(tmp_path), port=9000, tries=3)
