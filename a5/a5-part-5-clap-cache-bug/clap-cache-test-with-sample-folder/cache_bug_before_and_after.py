"""Compare this demo against local main and HEAD without changing the checkout."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[3]
DEMO = Path(__file__).with_name("debug_reindex.py")


def snapshot(ref, destination):
    # Resolve the ref at run time and copy only its committed Python packages.
    commit = subprocess.check_output(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
                                     cwd=ROOT, text=True).strip()
    archive = destination / "source.tar"
    subprocess.run(["git", "archive", "--format=tar", f"--output={archive}", commit,
                    "pipeline/vocalocate_pipeline", "demo/src/vocalocate_demo"], cwd=ROOT, check=True)
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")
    archive.unlink()
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("add", "remove", "replace", "reorder", "all"), default="all")
    args = parser.parse_args()
    scenarios = ("add", "remove", "replace", "reorder") if args.scenario == "all" else (args.scenario,)
    results = {}
    with tempfile.TemporaryDirectory(prefix="vocalocate-compare-") as tmp:
        # Local main is the before version; HEAD is the current branch's committed version.
        # This does not check out either ref or include uncommitted changes.
        for label, ref in (("BEFORE (local main)", "main"), ("AFTER (current HEAD)", "HEAD")):
            folder = Path(tmp) / label.split()[0].lower()
            folder.mkdir()
            commit = snapshot(ref, folder)
            print(f"\n=== {label}: {commit[:12]} ===", flush=True)
            env = os.environ.copy()
            # Each subprocess imports its temporary source copy instead of the checkout.
            env["PYTHONPATH"] = os.pathsep.join((str(folder / "demo/src"),
                                                  str(folder / "pipeline"),
                                                  env.get("PYTHONPATH", "")))
            for scenario in scenarios:
                print(f"\n--- {scenario} ---", flush=True)
                result = subprocess.run([sys.executable, "-u", str(DEMO), "--scenario", scenario],
                                        cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
                print(result.stdout, end="", flush=True)
                print(f"RESULT: {'PASS' if result.returncode == 0 else 'FAIL'}", flush=True)
                results[(ref, scenario)] = result.returncode
    if any(results[("HEAD", scenario)] for scenario in scenarios):
        return 1
    if any(not results[("main", scenario)] for scenario in scenarios):
        print("\nNote: at least one scenario also passed on local main; it may already include the fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
