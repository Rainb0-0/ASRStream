from __future__ import annotations

import json
import sys

from .contracts import TranscriptEvent, TranscriptSink


class NdjsonStdoutSink(TranscriptSink):
    """Stable command-line output adapter for downstream consumers."""

    def emit(self, event: TranscriptEvent) -> None:
        print(json.dumps({
            "type": "transcript",
            "stream_id": event.stream_id,
            "language": event.language,
            "segments": [
                {"start_seconds": segment.start_seconds, "end_seconds": segment.end_seconds, "text": segment.text}
                for segment in event.segments
            ],
            "processing_seconds": event.processing_seconds,
            "audio_seconds": event.audio_seconds,
            "rtf": event.processing_seconds / event.audio_seconds if event.audio_seconds else None,
            "overrun": event.overrun,
        }, ensure_ascii=False), flush=True)
