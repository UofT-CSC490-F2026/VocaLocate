"""MOSS inference wiring and command-line reporting, without loading model weights."""
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import moss


class Inputs(dict):
    def to(self, device):
        return self


class MossTests(unittest.TestCase):
    def test_bundled_prompt_keeps_task_and_output_sections(self):
        text = moss.DEFAULT_PROMPT_FILE.read_text(encoding='utf-8').strip()
        self.assertEqual(moss.load_prompt(), text)
        self.assertIn('TASK INSTRUCTIONS', text)
        self.assertIn('OUTPUT INSTRUCTIONS', text)
        self.assertIn('Acoustic Tags:', text)

    def test_arbitrary_prompt_file_is_sent_without_appended_instructions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'sfx'
            audio.mkdir()
            (audio / 'test.wav').touch()
            prompt = ('TASK INSTRUCTIONS\nDescribe audible properties.\n\nOUTPUT INSTRUCTIONS\n'
                      'Description: <source and acoustic qualities>\n'
                      'Acoustic Tags: <four descriptors>\nRaw model output: Reference only.')
            prompt_file = root / 'custom prompt.txt'
            prompt_file.write_text(prompt, encoding='utf-8')
            good = dict(status='success', tags=['rough'], description='Rough.')
            with patch.object(moss, 'load_runtime', return_value=(None, None, None)), \
                 patch.object(moss, 'generate_annotation', return_value=good) as annotate:
                status = moss.main([str(audio), '--prompt-file', str(prompt_file), '--no-report'])
            self.assertEqual(status, 0)
            self.assertEqual(annotate.call_args.args[1], prompt)
            self.assertNotIn('sharp, metallic', annotate.call_args.args[1])

    def test_inline_prompt_is_complete_and_empty_text_is_rejected(self):
        self.assertEqual(moss.load_prompt(text='  Full task and format.  '), 'Full task and format.')
        with self.assertRaises(ValueError):
            moss.load_prompt(text='   ')
        with self.assertRaises(ValueError):
            moss.load_prompt(Path('prompt.txt'), text='Both are specified')

    def test_invalid_prompt_files_fail_before_loading_model(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'test.wav').touch()
            empty = root / 'empty.txt'
            empty.write_text('  ', encoding='utf-8')
            for path in [empty, root / 'missing.txt']:
                with self.subTest(path=path), patch.object(moss, 'load_runtime') as load:
                    self.assertEqual(moss.main([str(root), '--prompt-file', str(path), '--no-report']), 2)
                    load.assert_not_called()

    def test_v1_response_labels_do_not_mix_raw_section_into_tags(self):
        raw = ('Description: An intermittent buzz with a rough texture.\n'
               'Acoustic Tags: low-pitched, buzzy, rough, intermittent\n'
               'Raw model output: For reference purpose only.\nDescription: Ignore this repetition.')
        result = moss.parse_annotation(raw)
        self.assertEqual(result['tags'], ['low-pitched', 'buzzy', 'rough', 'intermittent'])
        self.assertEqual(result['description'], 'An intermittent buzz with a rough texture.')
        self.assertEqual(result['temporal_pattern'], 'unknown')
        self.assertEqual(result['status'], 'success')

    def test_optional_auth_preserves_default_login_when_unset(self):
        with patch.dict(moss.os.environ, {}, clear=True):
            self.assertEqual(moss.huggingface_auth_kwargs(), {})
        with patch.dict(moss.os.environ, {'HF_TOKEN': '  '}, clear=True):
            self.assertEqual(moss.huggingface_auth_kwargs(), {})

    def test_pipeline_token_reaches_both_loaders_without_logging(self):
        model_class, processor_class = Mock(), Mock()
        torch = SimpleNamespace(float32='float32')
        modules = {
            'torch': torch,
            'src.modeling_moss_audio': SimpleNamespace(MossAudioModel=model_class),
            'src.processing_moss_audio': SimpleNamespace(MossAudioProcessor=processor_class),
        }
        output = StringIO()
        with patch.dict(sys.modules, modules), \
             patch.dict(moss.os.environ, {'HF_TOKEN': '  test-pipeline-token  '}, clear=True), \
             patch.object(moss, 'select_device', return_value='cpu'), redirect_stdout(output):
            moss.load_runtime(moss.MODEL_ID, 'cpu')
        self.assertEqual(model_class.from_pretrained.call_args.kwargs['token'], 'test-pipeline-token')
        self.assertEqual(processor_class.from_pretrained.call_args.kwargs['token'], 'test-pipeline-token')
        self.assertNotIn('test-pipeline-token', output.getvalue())

    def test_native_inputs_mask_dtype_and_output_slice(self):
        input_ids = MagicMock()
        input_ids.shape = (1, 10)
        features = Mock()
        processor = Mock(audio_token_id=123)
        processor.return_value = Inputs(input_ids=input_ids, audio_data=features)
        processor.decode.return_value = ' Description: Test. Tags: rough, low. Temporal pattern: steady '
        model = Mock(device='cpu', dtype='float32')
        model.generate.return_value = MagicMock()
        torch = SimpleNamespace(inference_mode=nullcontext)
        raw = moss.generate_response('prompt', model, processor, torch, audio=[0, 1])
        self.assertTrue(raw.startswith('Description:'))
        processor.assert_called_once_with(text='prompt', return_tensors='pt', audios=[[0, 1]])
        features.to.assert_called_once_with('float32')
        self.assertIn('audio_input_mask', model.generate.call_args.kwargs)
        self.assertFalse(model.generate.call_args.kwargs['do_sample'])
        self.assertEqual(model.generate.return_value.__getitem__.call_args.args[0], (0, slice(10, None)))

    def test_full_recording_and_optional_cap(self):
        loader = Mock(return_value=list(range(650)))
        processor = SimpleNamespace(config=SimpleNamespace(mel_sr=10))
        with patch.dict(sys.modules, {'src.audio_io': SimpleNamespace(load_audio=loader)}), \
             patch.object(moss, 'annotate_audio', return_value={'status': 'success', 'tags': ['low', 'rough']}) as annotate:
            result = moss.generate_annotation(Path('test.wav'), 'prompt', None, None, processor, None)
            self.assertEqual(len(annotate.call_args.args[0]), 650)
            self.assertEqual(result['analyzed_sec'], 65)
            result = moss.generate_annotation(Path('test.wav'), 'prompt', 12, None, processor, None)
            self.assertEqual(len(annotate.call_args.args[0]), 120)
            self.assertEqual(result['duration_sec'], 65)
            self.assertEqual(result['analyzed_sec'], 12)

    def test_grounded_text_recovery(self):
        with patch.object(moss, 'generate_response', side_effect=[
            'A deep, rumbling sound.', 'deep, rumbling, metallic'
        ]) as generate:
            result = moss.annotate_audio([0], 'prompt', None, None, None)
        self.assertEqual(result['tags'], ['deep', 'rumbling'])
        self.assertEqual(result['status'], 'success')
        self.assertEqual(generate.call_args_list[0].kwargs['audio'], [0])
        self.assertIsNone(generate.call_args.kwargs['audio'])
        self.assertIn('Text-only tag extraction', result['raw_model_output'])

    def test_custom_recovery_instructions_are_used_with_text_only(self):
        instructions = 'TASK INSTRUCTIONS\nExtract only supported texture words.\nOUTPUT INSTRUCTIONS\nReturn comma-separated descriptors.'
        with patch.object(moss, 'generate_response', side_effect=[
            'A rough, rattling sound.', 'rough, rattling, metallic'
        ]) as generate:
            result = moss.annotate_audio([0], 'audio instructions', None, None, None,
                                         recovery_prompt=instructions)
        self.assertEqual(generate.call_args.args[0], instructions + '\n\nMODEL RESPONSE TO PROCESS\nA rough, rattling sound.')
        self.assertIsNone(generate.call_args.kwargs['audio'])
        self.assertEqual(result['tags'], ['rough', 'rattling'])

    def test_cli_loads_custom_recovery_file_before_inference(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'test.wav').touch()
            prompt_file = root / 'recovery.txt'
            instructions = 'Extract acoustic descriptors from the supplied text only.'
            prompt_file.write_text(instructions, encoding='utf-8')
            with patch.object(moss, 'load_runtime', return_value=(None, None, None)), \
                 patch.object(moss, 'generate_annotation', return_value=dict(status='success', tags=['rough'])) as annotate:
                self.assertEqual(moss.main([str(root), '--recovery-prompt-file', str(prompt_file)]), 0)
            self.assertEqual(annotate.call_args.kwargs['recovery_prompt'], instructions)
            prompt_file.write_text(' ', encoding='utf-8')
            with patch.object(moss, 'load_runtime') as load:
                self.assertEqual(moss.main([str(root), '--recovery-prompt-file', str(prompt_file)]), 2)
                load.assert_not_called()

    def test_existing_tags_skip_recovery(self):
        with patch.object(moss, 'generate_response', return_value=(
            'Description: A metallic ringing sound. Tags: metallic, ringing. Temporal pattern: fading'
        )) as generate:
            result = moss.annotate_audio([0], 'prompt', None, None, None)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(result['tags'], ['metallic', 'ringing'])

    def test_cli_recurses_continues_after_errors_and_writes_only_html(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'sfx'
            audio.mkdir()
            (audio / 'a.wav').touch()
            (audio / 'nested').mkdir()
            (audio / 'nested' / 'b.WAV').touch()
            report_name = root / 'review.html'
            good = dict(status='success', tags=['rough', 'low'], description='A rough sound.',
                        temporal_pattern='steady', raw_model_output='example', analyzed_sec=1)
            with patch.object(moss, 'load_runtime', return_value=(None, None, None)), \
                 patch.object(moss, 'generate_annotation', side_effect=[good, ValueError('bad audio')]):
                status = moss.main([str(audio), '--html-report', str(report_name)])
            self.assertEqual(status, 1)
            self.assertEqual(sorted(p.name for p in root.iterdir()), ['review.html', 'sfx'])
            html = (root / 'review.html').read_text()
            self.assertIn('MOSS-Audio-4B-Instruct', html)
            self.assertIn('bad audio', html)
            self.assertIn('<td>success</td>', html)
            self.assertIn('<td>fail</td>', html)
            self.assertIn('<pre>example</pre>', html)
            self.assertNotIn('<details>', html)
            self.assertNotIn('<summary>', html)
            self.assertEqual(html.count('<tr>'), 3)
            with patch.object(moss, 'load_runtime') as load:
                self.assertEqual(moss.main([str(audio), '--html-report', str(report_name)]), 2)
                load.assert_not_called()

    def test_interruption_saves_completed_rows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a.wav').touch()
            (root / 'b.wav').touch()
            good = dict(status='success', tags=['rough'], description='Rough.', temporal_pattern='steady')
            with patch.object(moss, 'load_runtime', return_value=(None, None, None)), \
                 patch.object(moss, 'generate_annotation', side_effect=[good, KeyboardInterrupt]):
                with self.assertRaises(KeyboardInterrupt):
                    moss.main([str(root), '--html-report', str(root / 'partial.html')])
            html = (root / 'partial.html').read_text()
            self.assertIn('a.wav', html)
            self.assertNotIn('b.wav', html)

    def test_no_report_runs_without_html_module_and_writes_no_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a.wav').touch()
            good = dict(status='success', tags=['rough', 'low'], description='Rough.', temporal_pattern='steady')
            with patch.dict(sys.modules, {'html_report': None}), \
                 patch.object(moss, 'load_runtime', return_value=(None, None, None)), \
                 patch.object(moss, 'generate_annotation', return_value=good):
                status = moss.main([str(root)])
            self.assertEqual(status, 0)
            self.assertEqual([path.name for path in root.iterdir()], ['a.wav'])

    def test_inference_returns_dictionary_without_html_module(self):
        loader = Mock(return_value=[0] * 100)
        processor = SimpleNamespace(config=SimpleNamespace(mel_sr=10))
        with patch.dict(sys.modules, {'src.audio_io': SimpleNamespace(load_audio=loader), 'html_report': None}), \
             patch.object(moss, 'annotate_audio', return_value=dict(status='success', tags=['rough'])):
            annotation = moss.generate_annotation(Path('test.wav'), 'prompt', None, None, processor, None)
        self.assertIsInstance(annotation, dict)
        self.assertEqual(annotation['tags'], ['rough'])
        self.assertEqual(annotation['analyzed_sec'], 10)


if __name__ == '__main__':
    unittest.main()
