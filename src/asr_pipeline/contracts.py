from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol

import numpy as np


@dataclass(frozen=True)
class AudioWindow:
    stream_id: str
    start_seconds: float
    end_seconds: float
    samples: np.ndarray


@dataclass(frozen=True)
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True)
class TranscriptEvent:
    stream_id: str
    language: str | None
    segments: tuple[TranscriptSegment, ...]
    processing_seconds: float
    audio_seconds: float
    overrun: bool


class AudioSource(Protocol):
    def windows(self) -> AsyncIterator[AudioWindow]: ...


class TranscriptSink(Protocol):
    def emit(self, event: TranscriptEvent) -> None: ...
