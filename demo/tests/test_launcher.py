"""The desktop launcher (desktop.py) on every platform: data dir, GPU detection, single instance, backend
start, app window, message boxes, autostart and shortcut paths, and the headless server's lifecycle.

Platform branches run on any OS: `sys.platform` and the OS modules (msvcrt, fcntl, ctypes.windll, subprocess)
are replaced with fakes, so these tests never open a window, start a process or touch the real Start menu.
"""

import json
import logging
import os
import runpy
import socket
import subprocess
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path

import pytest

from vocalocate_demo import app as app_mod
from vocalocate_demo import cli, desktop


class FakeProc:
    def __init__(self, codes=(None,)):
        self.codes, self.returncode = list(codes), None

    def poll(self):
        self.returncode = self.codes.pop(0) if len(self.codes) > 1 else self.codes[0]
        return self.returncode


@pytest.fixture
def restore_logging():
    """serve_main reconfigures the root logger (basicConfig(force=True)); put pytest's handlers back."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for h in list(root.handlers):
        if h not in handlers:
            root.removeHandler(h)
            h.close()
    for h in handlers:
        if h not in root.handlers:
            root.addHandler(h)
    root.setLevel(level)


# -- data dir -------------------------------------------------------------------------------------------
def test_home_on_each_platform(monkeypatch, tmp_path):
    monkeypatch.delenv("VOCALOCATE_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert desktop.home() == tmp_path / "local" / "VocaLocate"
    monkeypatch.setattr(sys, "platform", "darwin")
    assert desktop.home() == Path.home() / "Library" / "Application Support" / "VocaLocate"
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert desktop.home() == tmp_path / "xdg" / "vocalocate"
    monkeypatch.delenv("XDG_DATA_HOME")
    assert desktop.home() == Path.home() / ".local" / "share" / "vocalocate"


def test_atomic_write_leaves_no_temp_file_on_failure(monkeypatch, tmp_path):
    def fail(*a):
        raise OSError("disk full")

    monkeypatch.setattr(desktop.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        desktop.save_settings(tmp_path, {"a": 1})
    assert list(tmp_path.iterdir()) == []


# -- GPU detection --------------------------------------------------------------------------------------
def _fresh_gpu(monkeypatch, platform_name, which=None, run=None):
    monkeypatch.setattr(desktop, "_gpu_cache", None)
    monkeypatch.setattr(sys, "platform", platform_name)
    monkeypatch.setattr(desktop.shutil, "which", lambda name: which)
    if run is not None:
        monkeypatch.setattr(desktop.subprocess, "run", run)


def test_gpu_info_windows_reads_nvidia_smi_and_caches(monkeypatch, tmp_path):
    (tmp_path / "System32").mkdir()
    (tmp_path / "System32" / "nvcuda.dll").write_bytes(b"")
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="NVIDIA GeForce RTX 5080\n")

    _fresh_gpu(monkeypatch, "win32", which="nvidia-smi", run=run)
    assert desktop.gpu_info() == {"available": True, "name": "NVIDIA GeForce RTX 5080"}
    assert desktop.gpu_info() is desktop.gpu_info() and len(calls) == 1  # cached


def test_gpu_info_nvidia_smi_failures_keep_the_gpu_unnamed(monkeypatch, tmp_path):
    import ctypes.util

    monkeypatch.setattr(ctypes.util, "find_library", lambda name: "libcuda.so.1")

    def broken(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 5)

    _fresh_gpu(monkeypatch, "linux", which="nvidia-smi", run=broken)
    assert desktop.gpu_info() == {"available": True, "name": None}
    _fresh_gpu(monkeypatch, "linux", which="nvidia-smi",
               run=lambda cmd, **kw: types.SimpleNamespace(returncode=9, stdout=""))
    assert desktop.gpu_info() == {"available": True, "name": None}


def test_gpu_info_linux_without_cuda_and_apple_silicon(monkeypatch):
    import ctypes.util
    import platform

    monkeypatch.setattr(ctypes.util, "find_library", lambda name: None)
    monkeypatch.setattr(desktop, "Path", lambda *a: types.SimpleNamespace(exists=lambda: False))
    _fresh_gpu(monkeypatch, "linux")
    assert desktop.gpu_info() == {"available": False, "name": None}
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    _fresh_gpu(monkeypatch, "darwin", which="nvidia-smi", run=lambda *a, **k: pytest.fail("ran nvidia-smi on a Mac"))
    assert desktop.gpu_info() == {"available": True, "name": "Apple GPU (Metal)"}


# -- single instance ------------------------------------------------------------------------------------
def test_instance_lock_is_exclusive_and_released_on_close(tmp_path):
    first = desktop.acquire_instance_lock(tmp_path)
    assert first is not None
    assert desktop.acquire_instance_lock(tmp_path) is None  # a second backend for the same data dir
    first.close()
    again = desktop.acquire_instance_lock(tmp_path)
    assert again is not None
    again.close()


@pytest.mark.parametrize("platform_name", ["win32", "linux"])
def test_instance_lock_uses_the_platform_lock(monkeypatch, tmp_path, platform_name):
    held = set()

    def lock(fd, *args):
        if "held" in held:
            raise OSError("locked")
        held.add("held")

    msvcrt = types.SimpleNamespace(LK_NBLCK=2, locking=lambda fd, mode, n: lock(fd, mode, n))
    fcntl = types.SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=lambda fd, flags: lock(fd, flags))
    monkeypatch.setitem(sys.modules, "msvcrt", msvcrt)
    monkeypatch.setitem(sys.modules, "fcntl", fcntl)
    monkeypatch.setattr(sys, "platform", platform_name)
    first = desktop.acquire_instance_lock(tmp_path)
    try:
        assert first is not None and desktop.acquire_instance_lock(tmp_path) is None
    finally:
        first.close()


def test_port_free_uses_exclusive_bind_on_windows(monkeypatch):
    opts = []

    class FakeSocket:
        def __init__(self, *a):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def setsockopt(self, *a):
            opts.append(a)

        def bind(self, addr):
            if addr[1] == 1:
                raise OSError("in use")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(socket, "SO_EXCLUSIVEADDRUSE", -5, raising=False)
    monkeypatch.setattr(desktop.socket, "socket", FakeSocket)
    assert desktop.port_free(2) is True and desktop.port_free(1) is False
    assert opts and opts[0][1] == -5


def test_running_port_needs_an_int_port(tmp_path):
    (tmp_path / "server.json").write_text(json.dumps({"pid": 1, "port": "8765"}), encoding="utf-8")
    assert desktop.running_port(tmp_path) is None


# -- backend --------------------------------------------------------------------------------------------
def test_pythonw_is_preferred_on_windows(monkeypatch, tmp_path):
    exe = tmp_path / "python.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setattr(sys, "platform", "win32")
    assert desktop._pythonw() == str(exe)  # no pythonw.exe next to it
    (tmp_path / "pythonw.exe").write_bytes(b"")
    assert desktop._pythonw() == str(tmp_path / "pythonw.exe")
    assert desktop.backend_command()[1:] == ["-m", "vocalocate_demo.desktop", "--serve"]


@pytest.mark.parametrize("exe, hidden", [("pythonw.exe", False), ("python.exe", True)])
def test_start_backend_on_windows_hides_the_console(monkeypatch, tmp_path, exe, hidden):
    seen = {}
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(desktop, "backend_command", lambda: [rf"C:\Py\{exe}", "-m", "vocalocate_demo.desktop", "--serve"])
    monkeypatch.setattr(desktop.subprocess, "Popen", lambda cmd, **kw: seen.update(cmd=cmd, **kw) or "proc")
    assert desktop.start_backend(tmp_path) == "proc"
    assert seen["creationflags"] == (0x200 | 0x08000000 if hidden else 0x200)
    assert seen["env"]["VOCALOCATE_HOME"] == str(tmp_path) and seen["stdout"].closed  # the child keeps its copy
    assert (tmp_path / "logs" / "server.log").exists()


def test_start_backend_elsewhere_starts_a_new_session(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(desktop.subprocess, "Popen", lambda cmd, **kw: seen.update(kw) or "proc")
    desktop.start_backend(tmp_path)
    assert seen["start_new_session"] is True and "creationflags" not in seen


def test_ensure_backend_reuses_starts_and_waits(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop.time, "sleep", lambda s: None)
    monkeypatch.setattr(desktop, "running_port", lambda h: 9001)
    monkeypatch.setattr(desktop, "start_backend", lambda h: pytest.fail("started a second backend"))
    assert desktop.ensure_backend(tmp_path) == 9001

    answers = [None, None, None, 9002]
    monkeypatch.setattr(desktop, "running_port", lambda h: answers.pop(0))
    # Exit code 0 means another backend holds the lock: keep waiting for its server.json.
    monkeypatch.setattr(desktop, "start_backend", lambda h: FakeProc([None, 0]))
    assert desktop.ensure_backend(tmp_path) == 9002


def test_ensure_backend_reports_a_crash_and_a_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop.time, "sleep", lambda s: None)
    monkeypatch.setattr(desktop, "running_port", lambda h: None)
    monkeypatch.setattr(desktop, "start_backend", lambda h: FakeProc([3]))
    with pytest.raises(RuntimeError, match=r"exited \(code 3\).*server.log"):
        desktop.ensure_backend(tmp_path)
    monkeypatch.setattr(desktop, "start_backend", lambda h: FakeProc([None]))
    with pytest.raises(RuntimeError, match="did not start in 0 s"):
        desktop.ensure_backend(tmp_path, timeout=0.0)


# -- app window -----------------------------------------------------------------------------------------
def test_browser_candidates_windows_mac_and_path(monkeypatch, tmp_path):
    # Built exactly as desktop.py joins them (backslash-separated): nested folders on Windows, and
    # one file whose name contains backslashes elsewhere.
    edge = tmp_path / "pf86" / r"Microsoft\Edge\Application\msedge.exe"
    chrome = tmp_path / "local" / r"Google\Chrome\Application\chrome.exe"
    for exe in (edge, chrome):
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"")
    monkeypatch.setenv("PROGRAMFILES(X86)", str(tmp_path / "pf86"))
    monkeypatch.delenv("PROGRAMFILES", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setattr(desktop.shutil, "which", lambda name: str(edge) if name == "msedge" else None)
    monkeypatch.setattr(sys, "platform", "win32")
    assert desktop._browser_candidates() == [str(edge), str(chrome)]  # Edge first, no duplicates

    mac_chrome = tmp_path / "Applications" / "Google Chrome.app" / "Contents" / "MacOS" / "Google Chrome"
    mac_chrome.parent.mkdir(parents=True)
    mac_chrome.write_bytes(b"")
    monkeypatch.setattr(desktop, "MAC_APPLICATIONS", tmp_path / "Applications")
    monkeypatch.setattr(desktop.shutil, "which", lambda name: "/usr/bin/chromium" if name == "chromium" else None)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert desktop._browser_candidates() == [str(mac_chrome), "/usr/bin/chromium"]


def test_open_window_tries_each_browser_then_the_default_one(monkeypatch, tmp_path):
    import webbrowser

    started = []

    def popen(cmd, **kw):
        if cmd[0] == "broken":
            raise OSError("not executable")
        started.append(cmd)

    monkeypatch.setattr(desktop, "_browser_candidates", lambda: ["broken", "edge"])
    monkeypatch.setattr(desktop.subprocess, "Popen", popen)
    assert desktop.open_window("http://127.0.0.1:1/", tmp_path) == "edge"
    assert started[0][1] == "--app=http://127.0.0.1:1/" and f"--user-data-dir={tmp_path / 'browser'}" in started[0]

    opened = []
    monkeypatch.setattr(desktop, "_browser_candidates", lambda: [])
    monkeypatch.setattr(webbrowser, "open", opened.append)
    assert desktop.open_window("http://127.0.0.1:2/", tmp_path) == "webbrowser" and opened == ["http://127.0.0.1:2/"]


def test_launch_opens_the_window_or_shows_the_error(monkeypatch, tmp_path):
    windows, errors = [], []
    monkeypatch.setattr(desktop, "open_window", lambda url, h: windows.append(url))
    monkeypatch.setattr(desktop, "_show_error", errors.append)
    monkeypatch.setattr(desktop, "ensure_backend", lambda h: 8800)
    assert desktop.launch(tmp_path) == 0 and windows == ["http://127.0.0.1:8800/"]
    assert desktop.launch(tmp_path, window=False) == 0 and len(windows) == 1

    def down(h):
        raise RuntimeError("backend exited")

    monkeypatch.setattr(desktop, "ensure_backend", down)
    assert desktop.launch(tmp_path) == 1 and errors == ["backend exited"]


def _fake_windll(monkeypatch):
    import ctypes

    boxes = []
    user32 = types.SimpleNamespace(MessageBoxW=lambda hwnd, text, title, flags: boxes.append((text, title, flags)))
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(user32=user32), raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    return boxes


def test_show_error_and_info_use_a_message_box_without_a_console(monkeypatch, capsys):
    boxes = _fake_windll(monkeypatch)
    desktop._show_error("it broke")
    assert "it broke" in capsys.readouterr().err and boxes == [("it broke", "VocaLocate", 0x10)]
    monkeypatch.setattr(sys, "stderr", None)
    desktop._show_error("again")  # pythonw: no stderr at all
    desktop._show_info("printed")
    assert "printed" in capsys.readouterr().out and len(boxes) == 2
    monkeypatch.setattr(sys, "stdout", None)
    desktop._show_info("boxed")
    assert boxes[-1] == ("boxed", "VocaLocate", 0x40)


def test_message_boxes_never_raise(monkeypatch):
    import ctypes

    def broken(*a):
        raise OSError("no user32")

    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(user32=types.SimpleNamespace(MessageBoxW=broken)),
                        raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(sys, "stdout", None)
    desktop._show_error("x")
    desktop._show_info("y")


# -- autostart and shortcuts: default folders -----------------------------------------------------------
def test_default_windows_folders(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "me"))
    programs = tmp_path / "roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    assert desktop.autostart_supported() and desktop.shortcuts_supported()
    assert desktop.startup_dir() == programs / "Startup"
    assert desktop.programs_dir() == programs and desktop.desktop_dir() == tmp_path / "me" / "Desktop"
    assert desktop._autostart_paths(None)[0] == programs / "Startup" / "VocaLocate.vbs"
    assert desktop.autostart_enabled() is False


def test_default_folders_are_windows_only(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="only supported on Windows"):
        desktop._autostart_paths(None)
    assert desktop.autostart_enabled() is False
    with pytest.raises(RuntimeError, match="only created on Windows"):
        desktop.remove_shortcut()


def test_install_shortcut_runs_powershell(monkeypatch, tmp_path):
    runs = []

    def run(cmd, **kw):
        runs.append(cmd)
        (tmp_path / desktop.SHORTCUT_FILE).write_bytes(b"lnk")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(desktop.subprocess, "run", run)
    assert desktop.install_shortcut(tmp_path) == tmp_path / desktop.SHORTCUT_FILE
    assert runs[0][0] == "powershell.exe" and "$s.Save()" in runs[0][-1]

    (tmp_path / desktop.SHORTCUT_FILE).unlink()
    monkeypatch.setattr(desktop.subprocess, "run",
                        lambda cmd, **kw: types.SimpleNamespace(returncode=1, stdout="", stderr="access denied\n"))
    with pytest.raises(RuntimeError, match="could not create .*access denied"):
        desktop.install_shortcut(tmp_path)


def test_shortcut_command_installs_and_removes_start_menu_and_desktop(monkeypatch, tmp_path):
    made, info = [], []
    monkeypatch.setattr(desktop, "desktop_dir", lambda: tmp_path / "Desktop")
    monkeypatch.setattr(desktop, "install_shortcut", lambda folder=None: made.append(folder) or Path("x.lnk"))
    monkeypatch.setattr(desktop, "remove_shortcut", lambda folder=None: folder is None)
    monkeypatch.setattr(desktop, "_show_info", info.append)
    assert desktop.main(["--install-shortcut", "--desktop"]) == 0
    assert made == [None, tmp_path / "Desktop"] and info[-1].count("shortcut created") == 2
    assert desktop.main(["--remove-shortcut", "--desktop"]) == 0
    assert info[-1] == "Start menu shortcut removed\nDesktop shortcut was not installed"


# -- headless server --------------------------------------------------------------------------------------
def test_build_state_uses_the_data_dir(tmp_path):
    state = desktop.build_state(tmp_path, workers=1, text=False)
    assert state.cfg.lake_root == tmp_path / "lake" and state.home == tmp_path and state.source == "installed"
    assert state.text_auto is False and state.models_dir == tmp_path / "lake" / "models" / "installed"


def _wait_for(cond, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return False


def _health(port):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/health")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_serve_main_serves_restarts_and_releases_the_lock(monkeypatch, tmp_path, restore_logging):
    home = tmp_path / "home"
    # A copy of the environment: serve_main hides the GPUs (CUDA_VISIBLE_DEVICES=""), which on the real
    # environment would put every later test in this process on the CPU once CUDA initialises.
    env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "VOCALOCATE_DEVICE")}
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(sys, "stdout", None)  # as under pythonw
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(app_mod, "serve_free", lambda state, host, port: app_mod.serve(state, host, 0))
    started, exits, seen = [], [], {}
    monkeypatch.setattr(desktop, "start_backend", lambda h: started.append((h, os.environ.get("CUDA_VISIBLE_DEVICES"))))
    monkeypatch.setattr(desktop.os, "_exit", exits.append)
    real_build = desktop.build_state

    def build(h):
        seen["state"] = real_build(h, text=False)
        seen["cuda"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        return seen["state"]

    monkeypatch.setattr(desktop, "build_state", build)
    result = {}
    thread = threading.Thread(target=lambda: result.update(code=desktop.serve_main(home)), daemon=True)
    thread.start()
    assert _wait_for(lambda: desktop.read_server_info(home) is not None)
    port = desktop.read_server_info(home)["port"]
    assert _health(port)["ok"] is True
    assert seen["cuda"] == ""  # CPU mode hides the GPUs before torch is imported
    assert desktop.acquire_instance_lock(home) is None  # held while serving
    assert seen["state"].status()["settings"]["restart_supported"] is True

    seen["state"].restart_server()
    thread.join(60)
    assert not thread.is_alive() and result["code"] == 0
    assert started == [(home, None)] and exits == [0]  # the new backend decides on the GPU itself
    assert desktop.read_server_info(home) is None
    lock = desktop.acquire_instance_lock(home)  # released before the new backend starts
    assert lock is not None
    lock.close()
    assert "serving" in (home / "logs" / "server.log").read_text(encoding="utf-8")


def test_serve_main_stops_on_ctrl_c(monkeypatch, tmp_path, restore_logging):
    monkeypatch.setattr(os, "environ", {**os.environ, "CUDA_VISIBLE_DEVICES": ""})  # already set: left alone
    created = []

    def serve_free(state, host, port):
        server = app_mod.serve(state, host, 0)

        def interrupted():
            raise KeyboardInterrupt

        server.serve_forever = interrupted
        created.append(server)
        return server

    monkeypatch.setattr(app_mod, "serve_free", serve_free)
    monkeypatch.setattr(desktop, "start_backend", lambda h: pytest.fail("restarted after Ctrl+C"))
    real_build = desktop.build_state

    def build(h):
        state = real_build(h, text=False)
        state.start = lambda: None  # no boot thread: this test is only about the shutdown path
        return state

    monkeypatch.setattr(desktop, "build_state", build)
    assert desktop.serve_main(tmp_path, port=8123) == 0
    assert created and desktop.read_server_info(tmp_path) is None


def test_serve_main_exits_when_another_backend_runs(monkeypatch, tmp_path, restore_logging):
    monkeypatch.setattr(desktop, "build_state", lambda h: pytest.fail("built a second server"))
    lock = desktop.acquire_instance_lock(tmp_path)
    try:
        assert desktop.serve_main(tmp_path) == 0  # the lock is held
    finally:
        lock.close()
    monkeypatch.setattr(desktop, "running_port", lambda h: 8765)
    assert desktop.serve_main(tmp_path) == 0  # an older backend without the lock answers on its port
    assert "already running" in (tmp_path / "logs" / "server.log").read_text(encoding="utf-8")


def test_main_dispatches_and_sets_the_data_dir(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("VOCALOCATE_HOME", "unused")
    monkeypatch.setattr(desktop, "serve_main", lambda h, p: calls.append(("serve", h, p)) or 0)
    monkeypatch.setattr(desktop, "launch", lambda h, window: calls.append(("launch", h, window)) or 0)
    assert desktop.main(["--serve", "--home", str(tmp_path), "--port", "9000"]) == 0
    assert os.environ["VOCALOCATE_HOME"] == str(tmp_path)
    assert desktop.main(["--background"]) == 0 and desktop.main([]) == 0
    assert calls == [("serve", tmp_path, 9000), ("launch", None, False), ("launch", None, True)]


def test_running_the_module_calls_main(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["vocalocate", "--install-shortcut"])
    monkeypatch.setattr(sys, "platform", "linux")  # shortcuts are Windows-only: main() reports it and returns 1
    with pytest.warns(RuntimeWarning), pytest.raises(SystemExit) as exit_info:
        runpy.run_module("vocalocate_demo.desktop", run_name="__main__")
    assert exit_info.value.code == 1


# -- cli --------------------------------------------------------------------------------------------------
def test_cli_serve_runs_the_desktop_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop, "serve_main", lambda h, p: ("served", h, p))
    assert cli.main(["serve", "--home", str(tmp_path), "--port", "9100"]) == ("served", tmp_path, 9100)


def test_installed_copy_uses_the_bundled_config(monkeypatch, tmp_path):
    from vocalocate_demo.cli import _default_config, _parser

    fake = tmp_path / "site-packages" / "vocalocate_demo" / "cli.py"
    (fake.parent / "configs").mkdir(parents=True)
    args = _parser().parse_args(["generate", "--out", "x"])
    monkeypatch.setattr(cli, "__file__", str(fake))
    assert _default_config(args) is None  # neither file exists
    (fake.parent / "configs" / "demo.toml").write_text("", encoding="utf-8")
    assert _default_config(args) == fake.parent / "configs" / "demo.toml"
