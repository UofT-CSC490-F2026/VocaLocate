"""Run the browser-free JavaScript regression as part of the existing pytest suite."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_ui_refreshes_results_after_rescan():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the frontend regression")
    script = Path(__file__).with_suffix(".js")
    subprocess.run([node, str(script)], check=True, timeout=20)
