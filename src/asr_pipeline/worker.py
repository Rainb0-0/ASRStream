from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

import numpy as np

from .config import ModelConfig
from .contracts import TranscriptSegment


@dataclass(frozen=True)
class WorkerSettings:
    model: ModelConfig
    benchmark_audio: str | None
    core_set: tuple[int, ...] | None


@dataclass(frozen=True)
class WorkItem:
    stream_id: str
    sequence: int
    start_seconds: float
    end_seconds: float
    samples: np.ndarray


@dataclass(frozen=True)
class BenchmarkResult:
    worker_id: int
    rtf: float | None
    error: str | None


@dataclass(frozen=True)
class WorkerReady:
    worker_id: int
    error: str | None


@dataclass(frozen=True)
class WorkResult:
    worker_id: int
    stream_id: str
    sequence: int
    language: str | None
    segments: tuple[TranscriptSegment, ...]
    processing_seconds: float
    audio_seconds: float
    error: str | None


def _set_affinity(core_set: tuple[int, ...] | None) -> None:
    if core_set is None:
        return
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("CPU affinity is not supported by this platform")
    os.sched_setaffinity(0, set(core_set))


def configure_cuda_libraries(model: ModelConfig) -> None:
    """Expose pip-installed CUDA runtime libraries to CTranslate2's dynamic loader."""
    if model.device != "cuda":
        return
    site_packages = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    directories = [
        site_packages / "nvidia" / "cublas" / "lib",
        site_packages / "nvidia" / "cudnn" / "lib",
        site_packages / "nvidia" / "cuda_nvrtc" / "lib",
    ]
    missing = [str(directory) for directory in directories if not directory.is_dir()]
    if missing:
        raise RuntimeError("CUDA runtime libraries are missing from the virtual environment: " + ", ".join(missing))
    current = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join([*(str(directory) for directory in directories), current]).rstrip(":")


def worker_main(worker_id: int, settings: WorkerSettings, input_queue: Any, output_queue: Any) -> None:
    """Process entry point. Model construction and affinity remain process-local."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        _set_affinity(settings.core_set)
        configure_cuda_libraries(settings.model)
        from faster_whisper import WhisperModel
        from faster_whisper.audio import decode_audio

        model_args: dict[str, Any] = {
            "device": settings.model.device,
            "compute_type": settings.model.compute_type,
            "cpu_threads": settings.model.cpu_threads,
            "num_workers": 1,
        }
        if settings.model.device == "cuda":
            model_args["device_index"] = settings.model.device_index
        model = WhisperModel(str(settings.model.model_path), **model_args)
        if settings.benchmark_audio is None:
            output_queue.put(WorkerReady(worker_id, None))
            benchmark_audio = None
        else:
            benchmark_audio = decode_audio(str(Path(settings.benchmark_audio)), sampling_rate=16000)
        if benchmark_audio is None:
            pass
        else:
            benchmark_duration = len(benchmark_audio) / 16000
            if benchmark_duration <= 0:
                raise RuntimeError("benchmark audio contains no samples")
            started = time.monotonic()
            # Consume the generator: Faster-Whisper starts decoding lazily.
            list(model.transcribe(
                benchmark_audio,
                beam_size=settings.model.beam_size,
                vad_filter=settings.model.vad_filter,
            )[0])
            output_queue.put(BenchmarkResult(worker_id, (time.monotonic() - started) / benchmark_duration, None))
    except Exception as error:
        output_queue.put(BenchmarkResult(worker_id, None, str(error)))
        return

    while True:
        item = input_queue.get()
        if item is None:
            return
        assert isinstance(item, WorkItem)
        started = time.monotonic()
        try:
            segments, info = model.transcribe(
                item.samples,
                beam_size=settings.model.beam_size,
                vad_filter=settings.model.vad_filter,
            )
            output = tuple(
                TranscriptSegment(
                    start_seconds=item.start_seconds + segment.start,
                    end_seconds=item.start_seconds + segment.end,
                    text=segment.text.strip(),
                )
                for segment in segments
                if segment.text.strip()
            )
            output_queue.put(WorkResult(
                worker_id=worker_id,
                stream_id=item.stream_id,
                sequence=item.sequence,
                language=getattr(info, "language", None),
                segments=output,
                processing_seconds=time.monotonic() - started,
                audio_seconds=item.end_seconds - item.start_seconds,
                error=None,
            ))
        except Exception as error:
            output_queue.put(WorkResult(
                worker_id=worker_id,
                stream_id=item.stream_id,
                sequence=item.sequence,
                language=None,
                segments=(),
                processing_seconds=time.monotonic() - started,
                audio_seconds=item.end_seconds - item.start_seconds,
                error=str(error),
            ))
