"""Deterministic tests for spoken-word timing and the s2s scoring path."""

import json
import tempfile
import unittest
from types import SimpleNamespace
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from omnisteval.s2s import (
    aligned_words_to_hypothesis,
    playback_seconds,
    validate_manifest,
    validate_playback_segments,
    whisperx_transcriber,
)
from omnisteval.s2s_cli import run_s2s
from omnisteval.cli import build_parser


class SpeechTimingTests(unittest.TestCase):
    def test_playback_offset_and_word_end(self):
        row = {"source": "sample.wav", "playback_start_seconds": 2.0}
        hypothesis = aligned_words_to_hypothesis(
            row, [{"word": "Hello", "start": 0.0, "end": 0.5}]
        )
        self.assertEqual(hypothesis["prediction"], "Hello")
        self.assertEqual(hypothesis["elapsed"], [2500.0])
        self.assertNotIn("delays", hypothesis)  # Playback time is CA, not CU.

    def test_chinese_characters_keep_word_end_timing(self):
        row = {"source": "sample.wav", "playback_start_seconds": 1.0}
        hypothesis = aligned_words_to_hypothesis(
            row, [{"word": "你好", "start": 0.0, "end": 0.4},
                  {"word": " 世界", "start": 0.5, "end": 0.9}],
            char_level=True,
        )
        self.assertEqual(hypothesis["prediction"], "你好世界")
        self.assertEqual(hypothesis["elapsed"], [1400.0, 1400.0, 1900.0, 1900.0])

    def test_compressed_audio_with_playback_gap(self):
        row = {
            "source": "sample.wav",
            "playback_segments": [
                {"audio_start_seconds": 0, "audio_end_seconds": 1,
                 "playback_start_seconds": 2},
                {"audio_start_seconds": 1, "audio_end_seconds": 2,
                 "playback_start_seconds": 5},
            ],
        }
        validate_playback_segments(row["playback_segments"])
        self.assertEqual(playback_seconds(1.5, row), 5.5)
        hypothesis = aligned_words_to_hypothesis(
            row, [{"word": "hello", "start": 0.1, "end": 0.5},
                  {"word": "world", "start": 1.1, "end": 1.5}]
        )
        self.assertEqual(hypothesis["elapsed"], [2500.0, 5500.0])

    def test_unmapped_or_unaligned_audio_rejected(self):
        row = {"source": "sample.wav", "playback_segments": [
            {"audio_start_seconds": 0, "audio_end_seconds": 1,
             "playback_start_seconds": 0}
        ]}
        with self.assertRaises(ValueError):
            playback_seconds(2, row)
        with self.assertRaises(ValueError):
            aligned_words_to_hypothesis(row, [{"word": "hello"}])

    def test_last_word_alignment_rounding_is_clamped(self):
        row = {"source": "sample.wav", "playback_segments": [
            {"audio_start_seconds": 0, "audio_end_seconds": 1,
             "playback_start_seconds": 2}
        ]}
        self.assertEqual(playback_seconds(1.002, row), 3.0)
        with self.assertRaises(ValueError):
            playback_seconds(1.1, row)

    def test_duplicate_source_rejected(self):
        rows = [{"source": "same.wav", "target_audio_filepath": "unused.wav"}] * 2
        with self.assertRaises(ValueError):
            validate_manifest(rows, require_audio=False)

    def test_whisperx_asr_and_alignment_adapter(self):
        calls = []
        fake = SimpleNamespace(
            load_model=lambda *args, **kwargs: SimpleNamespace(
                transcribe=lambda audio, batch_size: {
                    "language": "en", "segments": [{"text": "hello"}]
                }
            ),
            load_align_model=lambda **kwargs: ("align_model", "metadata"),
            load_audio=lambda path: "audio_samples",
            align=lambda segments, model, metadata, audio, device,
            return_char_alignments: (
                calls.append((segments, model, audio, device)) or
                {"word_segments": [{"word": "hello", "start": 0, "end": 0.4}]}
            ),
        )
        with patch.dict("sys.modules", {"whisperx": fake}):
            transcribe = whisperx_transcriber("tiny", "en", "cpu", "float32", 1)
            words = transcribe("audio.wav")
        self.assertEqual(words[0]["word"], "hello")
        self.assertEqual(calls[0][1:], ("align_model", "audio_samples", "cpu"))

    def test_s2s_parser_default_quality_model(self):
        args = build_parser().parse_args([
            "s2s", "--manifest", "m.jsonl", "--speech_segmentation", "s.yaml",
            "--ref_sentences_file", "r.txt", "--source_sentences_file", "src.txt",
            "--output_folder", "out",
        ])
        self.assertEqual(args.comet_model, "Unbabel/XCOMET-XL")

    def test_prealigned_end_to_end_without_model_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({
                "source": "sample.wav", "target_audio_filepath": "unused.wav",
                "playback_start_seconds": 0.25,
            }) + "\n")
            words = root / "aligned.jsonl"
            words.write_text(json.dumps({
                "source": "sample.wav",
                "words": [{"word": "Hello", "start": 0, "end": 0.4},
                          {"word": "world", "start": 0.6, "end": 1.1}],
            }) + "\n")
            segmentation = root / "segmentation.json"
            segmentation.write_text(json.dumps([
                {"wav": "sample.wav", "offset": 0, "duration": 2.0}
            ]))
            references = root / "references.txt"
            references.write_text("Hello world\n")
            sources = root / "sources.txt"
            sources.write_text("Hallo Welt\n")
            args = Namespace(
                manifest=manifest, prealigned_words=words,
                speech_segmentation=str(segmentation),
                ref_sentences_file=str(references),
                source_sentences_file=str(sources),
                output_folder=root / "output", target_language="en",
                whisper_model="large-v3", device="cpu", compute_type="float32",
                batch_size=1, comet_model="Unbabel/XCOMET-XL",
                no_comet=True, no_latency=False, bleu_tokenizer="13a",
            )
            report = run_s2s(args)
            self.assertEqual(report["recordings"], 1)
            self.assertEqual(report["spoken_words"], 2)
            self.assertFalse(report["latency_alignment_warning"])
            hypothesis = json.loads((root / "output/spoken_hypotheses.jsonl").read_text())
            self.assertEqual(hypothesis["elapsed"], [650.0, 1350.0])
            self.assertNotIn("delays", hypothesis)
            self.assertTrue((root / "output/instances.resegmented.jsonl").is_file())
            args.no_latency = True
            args.output_folder = root / "quality_only"
            quality_report = run_s2s(args)
            self.assertIsNone(quality_report["ASR-LongYAAL (playback, ms)"])
            self.assertEqual(quality_report["timestamp_convention"], "not scored")

    def test_chinese_end_to_end_uses_character_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.jsonl").write_text(json.dumps({
                "source": "sample.wav", "target_audio_filepath": "unused.wav",
                "playback_start_seconds": 1.0,
            }) + "\n")
            (root / "aligned.jsonl").write_text(json.dumps({
                "source": "sample.wav", "words": [
                    {"word": "你好", "start": 0, "end": 0.4},
                    {"word": "世界", "start": 0.5, "end": 0.9},
                ],
            }) + "\n")
            (root / "segmentation.json").write_text(json.dumps([
                {"wav": "sample.wav", "offset": 0, "duration": 2.0}
            ]))
            (root / "references.txt").write_text("你好世界\n")
            (root / "sources.txt").write_text("Hello world\n")
            args = Namespace(
                manifest=root / "manifest.jsonl",
                prealigned_words=root / "aligned.jsonl",
                speech_segmentation=str(root / "segmentation.json"),
                ref_sentences_file=str(root / "references.txt"),
                source_sentences_file=str(root / "sources.txt"),
                output_folder=root / "output", target_language="zh",
                whisper_model="large-v3", device="cpu", compute_type="float32",
                batch_size=1, comet_model="Unbabel/XCOMET-XL",
                no_comet=True, no_latency=False, bleu_tokenizer=None,
            )
            report = run_s2s(args)
            self.assertEqual(report["latency_unit"], "character")
            self.assertEqual(report["bleu_tokenizer"], "zh")
            self.assertEqual(report["spoken_units"], 4)
            hypothesis = json.loads((root / "output/spoken_hypotheses.jsonl").read_text())
            self.assertEqual(hypothesis["prediction"], "你好世界")
            self.assertEqual(hypothesis["elapsed"], [1400.0, 1400.0, 1900.0, 1900.0])


if __name__ == "__main__":
    unittest.main()
