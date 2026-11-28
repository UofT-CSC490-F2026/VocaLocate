"""Launch the historical pre-fix web UI without changing the current branch."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from cache_bug_before_and_after import BEFORE_COMMIT, ROOT, snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True, help="folder containing a compatible .pt checkpoint")
    parser.add_argument("--lake", type=Path, required=True, help="separate data-lake folder for this demo")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--clap-backend", choices=("hf", "fake"), default="hf")
    args = parser.parse_args()
    models, lake = args.models.expanduser().resolve(), args.lake.expanduser().resolve()
    if not models.is_dir():
        parser.error(f"models folder does not exist: {models}")
    if not any((models / f"{name}.pt").is_file() for name in ("shared", "metadata", "retrieval", "baseline", "random")):
        parser.error(f"no supported model checkpoint in {models}")
    with tempfile.TemporaryDirectory(prefix="vocalocate-before-ui-") as tmp:
        folder = Path(tmp)
        commit = snapshot(BEFORE_COMMIT, folder)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(folder / "demo/src"), str(folder / "pipeline"),
                                              env.get("PYTHONPATH", "")))
        print(f"Starting pre-fix commit {commit[:12]} at http://127.0.0.1:{args.port}", flush=True)
        print("Press Ctrl+C to stop the server and remove its temporary source copy.", flush=True)
        try:
            return subprocess.call([sys.executable, "-m", "vocalocate_demo.cli", "--lake", str(lake),
                                    "app", "--models", str(models), "--text", "--port", str(args.port),
                                    "--clap-backend", args.clap_backend], cwd=ROOT, env=env)
        except KeyboardInterrupt:
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
