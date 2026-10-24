"""Check source matching and reporting without dataset downloads or audio dependencies."""

import csv
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from scripts.esc50_data_parsing.report_vimsketch_overlap import build_report, FIELDS


SCRIPT = Path(__file__).resolve().parents[1] / "report_vimsketch_overlap.py"
CAT = "000Animal_Domestic animals_pets_Cat_Growling.wav"
VIS_CAT = "000Animal_Domestic animals_ pets_Cat_Growling_reference.wav"
DOG = "001Animal_Dog_Bark.wav"
SYNTH = "003-synth-commercial_synthesizers.wav"
MANIFEST_FIELDS = ("sfx_file", "recording_id")


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class OverlapReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "pairs.csv"
        self.metadata = self.root / "original_recordings_license.csv"
        self.vim = self.root / "VimSketch"
        self.vim.mkdir()
        (self.vim / "reference_file_names.csv").write_text(
            f"001_{CAT}\n002_{DOG}\n303_{SYNTH}\n", encoding="utf-8",
        )
        self.licence_rows = [
            {"filename": VIS_CAT, "freesound_id": "12"},
            {"filename": "001Animal_Dog_Bark_reference.wav", "freesound_id": ""},
            {"filename": "extra_VIS_candidate.wav", "freesound_id": "20"},
            {"filename": SYNTH, "freesound_id": "99"},
        ]
        write_csv(self.metadata, ("filename", "freesound_id"), self.licence_rows)
        self.pairs = [self.target(12), self.target(12), self.target(12, "B"),
                      self.target(20), self.target(99)]
        write_csv(self.manifest, MANIFEST_FIELDS, self.pairs)

    @staticmethod
    def target(fid, take="A"):
        target = f"1-{fid}-{take}-0"
        return {"sfx_file": f"audio/{target}.wav", "recording_id": f"freesound:{fid}"}

    def command(self, output, script=SCRIPT):
        return [sys.executable, str(script), "--manifest", str(self.manifest),
                "--vis-metadata", str(self.metadata), "--vimsketch-path", str(self.vim),
                "--out", str(output)]

    def test_collapses_repeated_pairs_preserves_distinct_takes_and_excludes_non_vis_inventory(self):
        rows, warnings, mapped, total = build_report(self.manifest, self.metadata, self.vim)
        self.assertEqual((mapped, total), (1, 2))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["esc50_filename"] for row in rows}, {"1-12-A-0.wav", "1-12-B-0.wav"})
        self.assertEqual({row["recording_id"] for row in rows}, {"freesound:12"})
        self.assertEqual(rows[0]["vimsketch_filename"], f"001_{CAT}")
        self.assertEqual(rows[0]["vis_filename"], VIS_CAT)
        self.assertEqual(len(warnings), 1)
        self.assertIn("overlap is unknown", warnings[0])

    def test_ambiguous_name_normalization_fails_instead_of_guessing(self):
        write_csv(self.metadata, ("filename", "freesound_id"), [*self.licence_rows,
                  {"filename": CAT, "freesound_id": "456"}])
        with self.assertRaisesRegex(ValueError, "Ambiguous VIS metadata"):
            build_report(self.manifest, self.metadata, self.vim)

    def test_missing_metadata_is_reported_as_unknown(self):
        write_csv(self.metadata, ("filename", "freesound_id"), self.licence_rows[1:])
        rows, warnings, mapped, total = build_report(self.manifest, self.metadata, self.vim)
        self.assertEqual((rows, mapped, total), ([], 0, 2))
        self.assertTrue(any("No VIS metadata match" in warning for warning in warnings))

    def test_audio_folder_names_work_without_filename_lists(self):
        path = self.vim / "reference_file_names.csv"
        names = path.read_text().splitlines()
        path.unlink()
        directory = self.vim / "references"
        directory.mkdir()
        for name in names:
            (directory / name).touch()
        (directory / "._hidden.wav").touch()
        rows, _, _, _ = build_report(self.manifest, self.metadata, self.vim)
        self.assertEqual(len(rows), 2)

    def test_cli_writes_csv_and_does_not_modify_inputs(self):
        before = {path: path.read_bytes() for path in (self.manifest, self.metadata,
                  self.vim / "reference_file_names.csv")}
        output = self.root / "nested" / "report.csv"
        result = subprocess.run(self.command(output), cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Affected ESC targets: 2", result.stdout)
        self.assertIn("Warning: No Freesound ID", result.stderr)
        with output.open(newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, ["recording_id", "esc50_filename",
                                               "vimsketch_filename", "vis_filename"])
            self.assertEqual(len(list(reader)), 2)
        for path, data in before.items():
            self.assertEqual(path.read_bytes(), data)

    def test_cli_preserves_existing_output_on_invalid_input_or_input_output_collision(self):
        output = self.root / "report.csv"
        output.write_text("previous output")
        write_csv(self.manifest, MANIFEST_FIELDS, [{**self.pairs[0], "recording_id": "unknown"}])
        result = subprocess.run(self.command(output), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_text(), "previous output")
        before = self.manifest.read_bytes()
        result = subprocess.run(self.command(self.manifest), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_default_report_location_is_beside_script_and_zero_matches_still_writes_header(self):
        tool = self.root / "tools"
        tool.mkdir()
        script = tool / SCRIPT.name
        shutil.copyfile(SCRIPT, script)
        write_csv(self.manifest, MANIFEST_FIELDS, [self.target(777)])
        args = self.command(self.root / "unused", script=script)[:-2]
        result = subprocess.run(args, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with (tool / "output" / "vimsketch_vis_overlaps.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, list(FIELDS))
            self.assertEqual(list(reader), [])


if __name__ == "__main__":
    unittest.main()
