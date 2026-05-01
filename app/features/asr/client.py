from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

LOGGER = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - handled at runtime
    WhisperModel = None  # type: ignore[assignment]


@dataclass(slots=True)
class TranscriptionSegment:
    text: str
    start: Optional[float]
    end: Optional[float]


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    language: Optional[str]
    duration: Optional[float]
    segments: List[TranscriptionSegment]


class ASRService:
    """Base interface for audio transcription backends."""

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:  # noqa: D401
        raise NotImplementedError


class FasterWhisperASR(ASRService):
    """faster-whisper based ASR implementation."""

    def __init__(
        self,
        model_size: str,
        *,
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 1,
        vad_filter: bool = True,
        language: str | None = None,
        condition_on_previous_text: bool = False,
    ) -> None:
        if WhisperModel is None:  # pragma: no cover - import guard
            raise RuntimeError("faster-whisper is not installed. Please install faster-whisper to enable ASR.")

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._beam_size = max(1, int(beam_size))
        self._vad_filter = bool(vad_filter)
        self._language = language
        self._condition_on_previous_text = condition_on_previous_text

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        path_str = str(audio_path)
        LOGGER.debug("Starting faster-whisper transcription: path=%s", path_str)
        segments_iter, info = self._model.transcribe(
            path_str,
            beam_size=self._beam_size,
            vad_filter=self._vad_filter,
            language=self._language,
            condition_on_previous_text=self._condition_on_previous_text,
        )

        segments: List[TranscriptionSegment] = []
        text_parts: List[str] = []

        for segment in segments_iter:
            cleaned = (segment.text or "").strip()
            if cleaned:
                text_parts.append(cleaned)
            segments.append(TranscriptionSegment(cleaned, getattr(segment, "start", None), getattr(segment, "end", None)))

        transcript = " ".join(text_parts).strip()

        LOGGER.debug(
            "faster-whisper finished transcription: language=%s duration=%s chars=%s",
            getattr(info, "language", None),
            getattr(info, "duration", None),
            len(transcript),
        )

        return TranscriptionResult(
            text=transcript,
            language=getattr(info, "language", None),
            duration=getattr(info, "duration", None),
            segments=segments,
        )


def create_asr_service(
    backend: str,
    *,
    model: str,
    device: str = "cuda",
    compute_type: str = "float16",
    beam_size: int = 1,
    vad_filter: bool = True,
    language: str | None = None,
    condition_on_previous_text: bool = False,
) -> ASRService:
    """Factory to build an ASR service from settings."""
    backend_normalized = backend.strip().lower()
    if backend_normalized == "faster_whisper":
        return FasterWhisperASR(
            model_size=model,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            vad_filter=vad_filter,
            language=language,
            condition_on_previous_text=condition_on_previous_text,
        )

    raise ValueError(f"Unsupported ASR backend: {backend}")


__all__ = [
    "ASRService",
    "FasterWhisperASR",
    "TranscriptionResult",
    "TranscriptionSegment",
    "create_asr_service",
]
