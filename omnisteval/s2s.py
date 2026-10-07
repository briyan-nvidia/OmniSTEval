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

"""Speech-output transcription and playback-timestamp conversion.

The resulting JSONL uses OmniSTEval's ordinary long-form hypothesis format.
Word *ends* are used: a spoken word is available to a listener only once it
has finished playing. Values are playback-clock milliseconds (CA), not CU.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def read_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_manifest(rows: Sequence[dict], require_audio: bool = True) -> None:
    if not rows:
        raise ValueError("Speech-to-speech manifest is empty")
    sources = set()
    for row in rows:
        source = row.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError("Every row needs a nonempty 'source' recording name")
        if source in sources:
            raise ValueError(f"Duplicate source recording: {source}")
        sources.add(source)
        audio = row.get("target_audio_filepath")
        if not isinstance(audio, str) or not audio:
            raise ValueError(f"Missing target_audio_filepath for {source}")
        if require_audio and not Path(audio).is_file():
            raise FileNotFoundError(audio)
        if "playback_start_seconds" in row:
            _nonnegative(row["playback_start_seconds"], "playback_start_seconds")
        if "playback_segments" in row:
            validate_playback_segments(row["playback_segments"])
        if "playback_segments" in row and "playback_start_seconds" in row:
            raise ValueError(f"Use only one playback timing convention for {source}")


def _nonnegative(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return number


def validate_playback_segments(segments: Sequence[dict]) -> None:
    if not segments:
        raise ValueError("playback_segments cannot be empty")
    previous_audio_end = 0.0
    previous_playback_end = 0.0
    for segment in segments:
        start = _nonnegative(segment.get("audio_start_seconds"), "audio_start_seconds")
        end = _nonnegative(segment.get("audio_end_seconds"), "audio_end_seconds")
        playback = _nonnegative(segment.get("playback_start_seconds"), "playback_start_seconds")
        if end <= start:
            raise ValueError("Each playback segment must have positive audio duration")
        if start < previous_audio_end - 1e-6 or playback < previous_playback_end - 1e-6:
            raise ValueError("Playback segments must be nonoverlapping and in playback order")
        previous_audio_end, previous_playback_end = end, playback + end - start


def playback_seconds(audio_seconds: float, row: dict) -> float:
    """Map a timestamp in the target-audio file onto the listener's clock."""
    time = _nonnegative(audio_seconds, "word audio time")
    segments = row.get("playback_segments")
    if segments is None:
        return time + float(row.get("playback_start_seconds", 0))
    for segment in segments:
        start = float(segment["audio_start_seconds"])
        end = float(segment["audio_end_seconds"])
        if start - 1e-6 <= time <= end + 1e-6:
            return float(segment["playback_start_seconds"]) + max(0.0, time - start)
    raise ValueError(f"Audio time {time:.3f}s has no playback mapping for {row['source']}")


def aligned_words_to_hypothesis(row: dict, words: Sequence[dict]) -> dict:
    """Create one OmniSTEval hypothesis from timed WhisperX words."""
    units: List[str] = []
    elapsed: List[float] = []
    previous_end = 0.0
    for item in words:
        word = item.get("word", "").strip()
        if not word:
            continue
        if "start" not in item or "end" not in item:
            raise ValueError(f"Unaligned word {word!r} in {row['source']}")
        start = _nonnegative(item["start"], "word start")
        end = _nonnegative(item["end"], "word end")
        if end < start or end < previous_end - 1e-6:
            raise ValueError(f"Nonmonotonic aligned words in {row['source']}")
        previous_end = end
        emitted = playback_seconds(end, row) * 1000.0
        for unit in word.split():
            units.append(unit)
            elapsed.append(emitted)
    hypothesis = {"source": row["source"], "prediction": " ".join(units), "elapsed": elapsed}
    if "source_length_ms" in row:
        hypothesis["source_length"] = _nonnegative(row["source_length_ms"], "source_length_ms")
    return hypothesis


def whisperx_transcriber(
    model_name: str, language: str, device: str, compute_type: str, batch_size: int
) -> Callable[[str], List[dict]]:
    """Create a reusable WhisperX ASR + forced-alignment callable.

    Imported lazily so text-only OmniSTEval does not require WhisperX.
    """
    try:
        import whisperx
    except ImportError as exc:
        raise ImportError("Install OmniSTEval[speech] to transcribe target audio") from exc
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)
    align_model, align_metadata = whisperx.load_align_model(language_code=language, device=device)

    def transcribe(audio_path: str) -> List[dict]:
        audio = whisperx.load_audio(audio_path)
        transcript = model.transcribe(audio, batch_size=batch_size)
        if transcript.get("language") not in (None, language):
            raise ValueError(f"Expected target language {language}, got {transcript['language']}")
        result = whisperx.align(
            transcript["segments"], align_model, align_metadata, audio, device,
            return_char_alignments=False,
        )
        return result["word_segments"]

    return transcribe


def build_hypotheses(
    manifest: Sequence[dict],
    transcribe: Optional[Callable[[str], List[dict]]] = None,
    prealigned: Optional[Sequence[dict]] = None,
) -> Tuple[List[dict], List[dict]]:
    """Return standard OmniSTEval hypotheses and a reusable alignment cache."""
    if (transcribe is None) == (prealigned is None):
        raise ValueError("Provide exactly one of transcribe or prealigned")
    aligned_by_source: Dict[str, dict] = {}
    if prealigned is not None:
        for record in prealigned:
            source = record.get("source")
            if source in aligned_by_source:
                raise ValueError(f"Duplicate prealigned source: {source}")
            aligned_by_source[source] = record
        if set(aligned_by_source) != {row["source"] for row in manifest}:
            raise ValueError("Prealigned source names must exactly match manifest sources")
    hypotheses = []
    alignments = []
    for row in manifest:
        words = (
            aligned_by_source[row["source"]]["words"]
            if prealigned is not None else transcribe(row["target_audio_filepath"])
        )
        hypotheses.append(aligned_words_to_hypothesis(row, words))
        alignments.append({"source": row["source"], "words": words})
    return hypotheses, alignments


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
