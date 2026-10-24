"""Filename and CSV checks; no dataset download or audio dependencies needed."""

import csv
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from scripts.esc50_data_parsing.parse_esc50_voice import add_metadata, build_manifest, parse_recording


SCRIPT = Path(__file__).resolve().parents[1] / "parse_esc50_voice.py"


class ParseESC50VoiceTests(unittest.TestCase):
    def test_default_output_is_relative_to_script_location_not_working_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tool_dir = root / "esc50_data_parsing"
            tool_dir.mkdir()
            script = tool_dir / SCRIPT.name
            shutil.copyfile(SCRIPT, script)
            voice = root / "voice" / "000"
            voice.mkdir(parents=True)
            (voice / "1-100032-A-0_000.wav").touch()
            result = subprocess.run(
                [sys.executable, str(script), "--path", str(voice.parent)],
                cwd=root, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            output = tool_dir / "output" / "raw_paired_manifest.csv"
            self.assertTrue(output.is_file())
            self.assertFalse((root / "output").exists())
            with output.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(next(csv.DictReader(stream))["imitator_id"], "esc50_voice:000")

    def test_target_name_and_speaker_are_preserved(self):
        root = Path("vocal_imitation")
        for name in ("1-100032-A-0_01.wav", "1-100032-A-0.wav_01.wav"):
            with self.subTest(name=name):
                row = parse_recording(root / "01" / name, root)
                self.assertEqual(row["sfx_file"], "audio/1-100032-A-0.wav")
                self.assertEqual(row["voice_file"], f"01/{name}")
                self.assertEqual(row["imitator_id"], "esc50_voice:01")
                self.assertEqual(row["recording_id"], "freesound:100032")

    def test_mismatched_speaker_and_empty_target_are_rejected(self):
        root = Path("vocal_imitation")
        for relative in ("01/1-100032-A-0_02.wav", "01/_01.wav"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                parse_recording(root / relative, root)

    def test_parent_folder_and_hidden_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            root = parent / "vocal_imitation"
            for speaker in ("02", "01"):
                folder = root / speaker
                folder.mkdir(parents=True)
                (folder / f"1-100032-A-0_{speaker}.wav").touch()
                (folder / f"._1-100032-A-0_{speaker}.wav").touch()
            resolved, rows = build_manifest(parent)
            self.assertEqual(resolved, root.resolve())
            self.assertEqual([r["imitator_id"] for r in rows], ["esc50_voice:01", "esc50_voice:02"])
            self.assertEqual(build_manifest(root), (resolved, rows))

    def test_cli_writes_csv_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "vocal_imitation"
            (root / "01").mkdir(parents=True)
            recording = root / "01" / "1-100032-A-0_01.wav"
            recording.write_bytes(b"filename-only fixture")
            output = Path(temp) / "output" / "manifest.csv"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--path", str(root), "--out", str(output)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows, [{
                "voice_file": "01/1-100032-A-0_01.wav",
                "imitator_id": "esc50_voice:01",
                "category": "", "sfx_file": "audio/1-100032-A-0.wav",
                "recording_id": "freesound:100032",
            }])
            self.assertEqual(recording.read_bytes(), b"filename-only fixture")
            self.assertIn("Wrote 1 recordings", result.stdout)

    def test_invalid_or_empty_dataset_does_not_write_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "vocal_imitation"
            (root / "01").mkdir(parents=True)
            output = Path(temp) / "manifest.csv"
            for malformed in (False, True):
                if malformed:
                    (root / "01" / "target_02.wav").touch()
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), "--path", str(root), "--out", str(output)],
                    capture_output=True, text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())

    def test_numeric_source_id_order_then_filename_then_numeric_speaker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = [
                "10/1-100-A-0_10.wav", "2/1-100-A-0_2.wav",
                "2/5-9-B-0_2.wav", "10/5-9-A-0_10.wav",
                "2/5-9-A-0_2.wav", "2/1-10-A-0_2.wav",
            ]
            for relative in files:
                path = root / relative
                path.parent.mkdir(exist_ok=True)
                path.touch()
            _, rows = build_manifest(root)
            self.assertEqual([r["voice_file"] for r in rows], [
                "2/5-9-A-0_2.wav", "10/5-9-A-0_10.wav", "2/5-9-B-0_2.wav",
                "2/1-10-A-0_2.wav", "2/1-100-A-0_2.wav", "10/1-100-A-0_10.wav",
            ])

    def test_cli_constructs_sfx_paths_without_checking_each_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            voice = root / "vocal_imitation"
            (voice / "01").mkdir(parents=True)
            (voice / "01" / "1-100032-A-0_01.wav").touch()
            esc50 = root / "ESC-50"
            (esc50 / "audio").mkdir(parents=True)
            (esc50 / "meta").mkdir()
            (esc50 / "meta" / "esc50.csv").write_text(
                "filename,category,src_file\n1-100032-A-0.wav,dog,100032\n", encoding="utf-8",
            )
            output = root / "pairs.csv"
            for sfx_root in (esc50, esc50 / "audio"):
                with self.subTest(sfx_root=sfx_root):
                    result = subprocess.run(
                        [sys.executable, str(SCRIPT), "--path", str(voice),
                         "--esc50-path", str(sfx_root), "--out", str(output)],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    with output.open(newline="", encoding="utf-8") as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(rows[0]["sfx_file"], "audio/1-100032-A-0.wav")
                    self.assertNotIn("match_status", rows[0])
                    self.assertEqual(rows[0]["category"], "dog")
                    self.assertEqual(rows[0]["recording_id"], "freesound:100032")
                    self.assertEqual(list(rows[0]), [
                        "voice_file", "sfx_file", "imitator_id", "category", "recording_id",
                    ])

    def test_metadata_keeps_distinct_takes_and_speakers_under_one_recording(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "esc50.csv"
            path.write_text(
                "filename,category,src_file\n"
                "1-100210-A-36.wav,vacuum_cleaner,100210\n"
                "1-100210-B-36.wav,vacuum_cleaner,100210\n", encoding="utf-8",
            )
            root = Path("vocal_imitation")
            rows = [parse_recording(root / speaker / f"1-100210-{take}-36_{speaker}.wav", root)
                    for take, speaker in (("A", "000"), ("A", "001"), ("B", "000"))]
            add_metadata(rows, path)
            self.assertEqual(len(rows), 3)
            self.assertEqual({r["recording_id"] for r in rows}, {"freesound:100210"})
            self.assertEqual(len({r["sfx_file"] for r in rows}), 2)
            self.assertEqual(rows[0]["imitator_id"], "esc50_voice:000")
            self.assertEqual(rows[1]["imitator_id"], "esc50_voice:001")
            self.assertEqual({r["category"] for r in rows}, {"vacuum_cleaner"})

    def test_cli_supports_metadata_without_sfx_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "000").mkdir()
            (root / "000" / "1-100032-A-0_000.wav").touch()
            metadata = root / "esc50.csv"
            metadata.write_text("filename,category,src_file\n1-100032-A-0.wav,dog,100032\n", encoding="utf-8")
            output = root / "manifest.csv"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--path", str(root), "--esc50-metadata", str(metadata),
                 "--out", str(output)], capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with output.open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["imitator_id"], "esc50_voice:000")
            self.assertEqual(row["category"], "dog")
            self.assertEqual(row["recording_id"], "freesound:100032")
            self.assertEqual(row["sfx_file"], "audio/1-100032-A-0.wav")
            self.assertNotIn("esc50_filename", row)

    def test_manifest_paths_resolve_from_their_dataset_roots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            voice = root / "vocal_imitations" / "000"
            voice.mkdir(parents=True)
            (voice / "1-100032-A-0_000.wav").touch()
            audio = root / "audio"
            audio.mkdir()
            (audio / "1-100032-A-0.wav").touch()
            voice_root, rows = build_manifest(root)
            self.assertEqual(voice_root, voice.parent.resolve())
            self.assertEqual(len(rows), 1)
            self.assertTrue((voice_root / rows[0]["voice_file"]).is_file())
            self.assertTrue((root / rows[0]["sfx_file"]).is_file())
            self.assertEqual(list(rows[0]), [
                "voice_file", "sfx_file", "imitator_id", "category", "recording_id",
            ])

    def test_missing_metadata_target_leaves_existing_output_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "000").mkdir()
            (root / "000" / "1-100032-A-0_000.wav").touch()
            metadata = root / "esc50.csv"
            metadata.write_text("filename,category,src_file\n", encoding="utf-8")
            output = root / "manifest.csv"
            output.write_text("previous output", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--path", str(root), "--esc50-metadata", str(metadata),
                 "--out", str(output)], capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("targets absent", result.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "previous output")


if __name__ == "__main__":
    unittest.main()
