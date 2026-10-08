# Copyright 2026 OmniSTEval contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI adapter from generated target speech to OmniSTEval long-form scoring."""

from __future__ import annotations

import json
import math
import gc
from pathlib import Path

from .io import dump_instances_jsonl, load_resegmentation_inputs
from .resegment import resegment
from .s2s import (
    build_hypotheses,
    read_jsonl,
    validate_manifest,
    whisperx_transcriber,
    write_jsonl,
)
from .scoring import evaluate_instances


def add_s2s_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "s2s",
        help="Score spoken translation with WhisperX, ASR-COMET-XL, and playback-aware ASR-LongYAAL.",
    )
    parser.add_argument("--manifest", type=Path, required=True,
                        help="JSONL: source, target_audio_filepath, and playback timing per recording.")
    parser.add_argument("--speech_segmentation", required=True,
                        help="Source-side YAML/JSON segmentation, as for longform evaluation.")
    parser.add_argument("--ref_sentences_file", required=True)
    parser.add_argument("--source_sentences_file", required=True)
    parser.add_argument("--output_folder", type=Path, required=True)
    parser.add_argument("--prealigned_words", type=Path,
                        help="Reuse cached WhisperX word_segments JSONL; avoids re-running ASR.")
    parser.add_argument("--target_language", default="en",
                        help="WhisperX language code and OmniSTEval target tokenizer (default: en).")
    parser.add_argument("--whisper_model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute_type", default="float16")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--comet_model", default="Unbabel/XCOMET-XL")
    parser.add_argument("--no_comet", action="store_true",
                        help="Skip the large XCOMET-XL model (useful for conversion/latency tests).")
    parser.add_argument("--no_latency", action="store_true",
                        help="Score speech quality only when the true playback timeline is unavailable.")
    parser.add_argument("--bleu_tokenizer", default=None,
                        help="SacreBLEU tokenizer (default: zh for Chinese, 13a otherwise).")


def run_s2s(args) -> dict:
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive")
    char_level = args.target_language == "zh"
    bleu_tokenizer = args.bleu_tokenizer or ("zh" if char_level else "13a")
    manifest = read_jsonl(args.manifest)
    for row in manifest:
        audio = row.get("target_audio_filepath")
        if audio and not Path(audio).is_absolute():
            row["target_audio_filepath"] = str((args.manifest.parent / audio).resolve())
    validate_manifest(manifest, require_audio=args.prealigned_words is None)

    if args.prealigned_words is not None:
        prealigned = read_jsonl(args.prealigned_words)
        transcribe = None
    else:
        prealigned = None
        transcribe = whisperx_transcriber(
            args.whisper_model, args.target_language, args.device,
            args.compute_type, args.batch_size,
        )
    hypotheses, alignments = build_hypotheses(
        manifest, transcribe, prealigned, char_level=char_level
    )
    if transcribe is not None:
        # XCOMET-XL is large: do not retain WhisperX and its aligner on the GPU
        # while loading the quality model.
        del transcribe
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    output = args.output_folder
    output.mkdir(parents=True, exist_ok=True)
    hypothesis_path = output / "spoken_hypotheses.jsonl"
    write_jsonl(hypothesis_path, hypotheses)
    write_jsonl(output / "aligned_words.jsonl", alignments)

    reference_words, hypothesis_words, segmentation, references = load_resegmentation_inputs(
        speech_segmentation=args.speech_segmentation,
        text_segmentation=None,
        ref_sentences_file=args.ref_sentences_file,
        hypothesis_file=str(hypothesis_path),
        hypothesis_format="jsonl",
        char_level=char_level,
        offset_delays=False,
        fix_emission_ca_flag=False,
    )
    with open(args.source_sentences_file, encoding="utf-8") as stream:
        source_sentences = [line.strip() for line in stream]
    if len(source_sentences) != len(references):
        raise ValueError("Source sentences and reference segments must have equal lengths")
    instances, instance_dicts = resegment(
        reference_words, hypothesis_words, segmentation, references,
        char_level=char_level, lang=args.target_language,
    )
    dump_instances_jsonl(instance_dicts, str(output))
    scores, _, _ = evaluate_instances(
        instances,
        is_longform=True,
        bleu_tokenizer=bleu_tokenizer,
        compute_latency=not args.no_latency,
        compute_comet=not args.no_comet,
        comet_model=args.comet_model,
        source_sentences=source_sentences,
    )
    latency_ms = scores.get("ca_long_yaal")
    if not args.no_latency and (latency_ms is None or not math.isfinite(latency_ms)):
        raise ValueError("ASR-LongYAAL is undefined: no aligned spoken words were scored")
    negative_aligned_words = sum(
        value < -1000.0
        for instance in instances
        for value in (instance.emission_ca or [])
    )
    empty_recordings = sum(not hypothesis["prediction"] for hypothesis in hypotheses)
    empty_reference_segments = sum(
        bool(instance.reference.strip()) and not instance.prediction.strip()
        for instance in instances
    )
    reference_unit_count = sum(
        len(reference.replace(" ", "")) if char_level else len(reference.split())
        for reference in references
    )
    spoken_unit_count = sum(len(hypothesis["elapsed"]) for hypothesis in hypotheses)
    report = {
        "ASR-LongYAAL (playback, ms)": latency_ms,
        "ASR-COMET-XL": scores.get("comet") if not args.no_comet else None,
        "ASR-BLEU": scores.get("bleu"),
        "ASR-chrF": scores.get("chrf"),
        "comet_model": args.comet_model if not args.no_comet else None,
        "whisper_model": args.whisper_model if args.prealigned_words is None else "prealigned cache",
        "timestamp_convention": (
            "not scored" if args.no_latency else
            "audible word end on playback clock; computation-aware"
        ),
        "recordings": len(manifest),
        "spoken_words": spoken_unit_count if not char_level else None,
        "spoken_to_reference_word_ratio": (
            spoken_unit_count / reference_unit_count if reference_unit_count and not char_level else None
        ),
        "spoken_units": spoken_unit_count,
        "spoken_to_reference_unit_ratio": (
            spoken_unit_count / reference_unit_count if reference_unit_count else None
        ),
        "latency_unit": "character" if char_level else "word",
        "bleu_tokenizer": bleu_tokenizer,
        "empty_recordings": empty_recordings,
        "empty_reference_segments": empty_reference_segments,
        "negative_aligned_words_over_1s": negative_aligned_words,
        "latency_alignment_warning": negative_aligned_words > 0 or empty_reference_segments > 0,
    }
    (output / "s2s_scores.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report
