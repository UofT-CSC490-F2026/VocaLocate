"""Branch-coverage gaps in the desktop app (app.py, desktop.py) that the main suites leave open on Linux CI.

Each test drives one decision the other way from the existing tests: a corrupt library entry, someone
else's server.json, a platform with no known browsers, a server bound to every interface, a library that
was replaced or has no index yet, and so on. Nothing here opens a window or starts a real backend.
"""

import http.client
import json
import os
import sys
import threading
from pathlib import Path

from vocalocate_pipeline.config import PipelineConfig

from vocalocate_demo import app as app_mod
from vocalocate_demo import desktop
from vocalocate_demo import train as T
from vocalocate_demo.app import AppState, Library
from vocalocate_demo.model import DualEncoder


def _state(home: Path | None, tmp_path: Path, **kw) -> AppState:
    kw.setdefault("text_auto", False)
    root = home or tmp_path
    return AppState(PipelineConfig(lake_root=root / "lake", workers=1), desktop.SOURCE, home=home,
                    autostart_dir=root / "fake_startup", **kw)


# -- desktop.py -------------------------------------------------------------------------------------------
def test_load_settings_drops_malformed_library_entries(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"libraries": [
        {"name": "sfx", "roots": ["/sounds"]},
        {"name": "", "roots": ["/x"]},       # no name
        {"name": "bad", "roots": "/x"},      # roots is not a list
        "not a dict",
    ]}), encoding="utf-8")
    assert desktop.load_settings(tmp_path)["libraries"] == [{"name": "sfx", "roots": ["/sounds"]}]


def test_clear_server_info_leaves_another_instances_file(tmp_path):
    info = tmp_path / "server.json"
    info.write_text(json.dumps({"pid": os.getpid() + 1, "port": 1}), encoding="utf-8")
    desktop.clear_server_info(tmp_path)
    assert info.exists()  # another backend wrote it; not ours to remove


def test_browser_candidates_on_linux_without_any_browser(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(desktop.shutil, "which", lambda name: None)
    assert desktop._browser_candidates() == []


def test_remove_shortcut_from_an_explicit_folder_on_any_platform(tmp_path):
    link = tmp_path / desktop.SHORTCUT_FILE
    link.write_bytes(b"")
    assert desktop.remove_shortcut(tmp_path) is True and not link.exists()
    assert desktop.remove_shortcut(tmp_path) is False


def test_show_info_without_a_console_off_windows_is_silent(monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "platform", "linux")
    desktop._show_info("nobody sees this")  # no console and no message box: nothing to do, no error


def test_restart_leaves_a_preset_gpu_setting_alone(monkeypatch, tmp_path):
    """CUDA_VISIBLE_DEVICES was set before serve_main, so the restart must not remove it."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    monkeypatch.setattr(os, "environ", env)
    started, exits = [], []
    monkeypatch.setattr(desktop, "start_backend", lambda h: started.append(os.environ.get("CUDA_VISIBLE_DEVICES")))
    monkeypatch.setattr(desktop.os, "_exit", exits.append)
    monkeypatch.setattr(desktop.logging, "basicConfig", lambda **kw: None)  # keep pytest's log handlers
    states = []

    class Server:
        server_address = ("127.0.0.1", 0)

        def serve_forever(self):
            states[0].restart_server()  # as /api/restart would

        def shutdown(self):
            pass

        def server_close(self):
            pass

    real_build = desktop.build_state
    monkeypatch.setattr(desktop, "build_state", lambda h: states.append(real_build(h, text=False)) or states[-1])
    monkeypatch.setattr(app_mod, "serve_free", lambda state, host, port: Server())
    monkeypatch.setattr(AppState, "start", lambda self: None)
    assert desktop.serve_main(tmp_path / "home") == 0
    assert started == [""] and exits == [0]


# -- app.py -----------------------------------------------------------------------------------------------
def test_boot_skips_replaced_libraries_and_queues_unindexed_ones(tmp_path, monkeypatch):
    state = _state(tmp_path / "home", tmp_path)
    replaced, fresh = Library(name="old", roots=[]), Library(name="new", roots=[])
    state.libraries.update(old=replaced, new=fresh)
    monkeypatch.setattr(app_mod, "load_engine", lambda: None)
    monkeypatch.setattr(state, "available_models", lambda: ["metadata"])
    monkeypatch.setattr(state, "model", lambda name: (None, None, "v1"))

    def attach(lib, name, version):  # while booting, the user re-adds "old" with other folders
        state.libraries["old"] = Library(name="old", roots=["/elsewhere"])

    monkeypatch.setattr(state, "_attach_cached", attach)
    monkeypatch.setattr(state, "_maybe_load_clap", lambda: None)
    queued = []
    monkeypatch.setattr(state, "_queue_index", queued.append)
    state._boot()
    assert queued == [fresh]  # the replaced library is left to its replacement
    assert fresh.state != "ready"  # no cached index: not marked ready


def test_attach_cached_with_no_cache_attaches_nothing(tmp_path, monkeypatch):
    state = _state(tmp_path / "home", tmp_path)
    lib = Library(name="sfx", roots=[])
    monkeypatch.setattr(app_mod, "cached_index", lambda *a, **k: None)
    state._attach_cached(lib, "metadata", "v1")
    assert lib.indexes == {}


def test_state_without_a_home_never_writes_settings(tmp_path):
    state = _state(None, tmp_path)
    state._save_settings()
    state.libraries["sfx"] = Library(name="sfx", roots=[])
    state.delete_library("sfx")
    assert "sfx" not in state.libraries and not (tmp_path / "settings.json").exists()


def test_model_loaded_by_another_thread_while_waiting_for_the_lock(tmp_path):
    state = _state(tmp_path / "home", tmp_path)
    loaded = ("model", "cfg", "v1")

    class LateDict(dict):
        def get(self, key, default=None):  # the lock-free check runs before the other thread finishes
            return None

    state.models = LateDict(metadata=loaded)
    assert state.model("metadata") is loaded  # found under the lock: not loaded twice


def test_reimport_of_the_same_version_keeps_complete_libraries(tmp_path, monkeypatch):
    state = _state(tmp_path / "home", tmp_path)
    src = tmp_path / "v1"
    T.save(DualEncoder(dim=16, width=4, frames=64), src / "metadata.pt", T.TrainConfig(frames=64, device="cpu"),
           {"model_version": "m1"})
    state.import_models(str(src))
    lib = Library(name="sfx", roots=[])
    lib.indexes["metadata"] = "built with m1"
    state.libraries["sfx"] = lib
    rescans = []
    monkeypatch.setattr(state, "rescan", rescans.append)
    state.import_models(str(state.models_dir))
    assert rescans == [] and lib.indexes == {"metadata": "built with m1"}


def test_server_bound_to_every_interface_skips_the_host_check(tmp_path):
    server = app_mod.serve(_state(tmp_path / "home", tmp_path), host="0.0.0.0", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=30)
        conn.request("GET", "/api/status", headers={"Host": "my-laptop.local"})
        assert conn.getresponse().status == 200  # LAN access is the point of binding 0.0.0.0
        conn.close()
    finally:
        server.shutdown()
        server.server_close()


