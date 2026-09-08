from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
import multiprocessing as mp
from queue import Empty
import sys

from .config import PipelineConfig
from .contracts import AudioWindow, TranscriptEvent, TranscriptSink
from .source import FfmpegAudioSource
from .worker import BenchmarkResult, WorkItem, WorkResult, WorkerSettings, configure_cuda_libraries, worker_main


class CapacityError(RuntimeError):
    """Raised when declared resources cannot maintain the configured sources."""


def capacity_for_rtf(rtf: float) -> int:
    """Conventional RTF is wall time divided by audio time."""
    if rtf <= 0 or not math.isfinite(rtf):
        raise CapacityError(f"invalid benchmark RTF: {rtf}")
    if rtf >= 1:
        raise CapacityError(f"benchmark RTF is {rtf:.3f}; it must be below 1 for live transcription")
    return math.floor(1 / rtf)


def required_worker_count(stream_count: int, capacity: int) -> int:
    if capacity < 1:
        raise CapacityError("a worker has no live-stream capacity")
    return math.ceil(stream_count / capacity)


@dataclass
class WorkerHandle:
    worker_id: int
    process: mp.Process
    input_queue: mp.Queue
    capacity: int


@dataclass
class StreamState:
    worker: WorkerHandle
    pending: AudioWindow | None = None
    inflight_sequence: int | None = None
    next_sequence: int = 0


class Pipeline:
    def __init__(self, config: PipelineConfig, sink: TranscriptSink) -> None:
        self._config = config
        self._sink = sink
        self._context = mp.get_context("spawn")
        self._results: mp.Queue = self._context.Queue()
        self._workers: list[WorkerHandle] = []
        self._states: dict[str, StreamState] = {}
        self._stopping = False

    def _start_worker(self, worker_id: int) -> WorkerHandle:
        configure_cuda_libraries(self._config.model)
        core_set = self._config.workers.core_sets[worker_id] if self._config.workers.core_sets else None
        input_queue = self._context.Queue()
        settings = WorkerSettings(self._config.model, str(self._config.benchmark_audio), core_set)
        process = self._context.Process(target=worker_main, args=(worker_id, settings, input_queue, self._results), daemon=True)
        process.start()
        return WorkerHandle(worker_id, process, input_queue, capacity=0)

    def _await_benchmark(self, expected_ids: set[int]) -> dict[int, float]:
        measured: dict[int, float] = {}
        while expected_ids:
            try:
                message = self._results.get(timeout=120)
            except Empty as error:
                raise CapacityError("timed out waiting for Faster-Whisper startup benchmark") from error
            if not isinstance(message, BenchmarkResult) or message.worker_id not in expected_ids:
                continue
            expected_ids.remove(message.worker_id)
            if message.error:
                raise CapacityError(f"worker {message.worker_id} benchmark failed: {message.error}")
            assert message.rtf is not None
            measured[message.worker_id] = message.rtf
        return measured

    def start(self) -> None:
        # Retain the benchmarked model as worker zero, avoiding a duplicate large-v3 load.
        first = self._start_worker(0)
        self._workers.append(first)
        rtf = self._await_benchmark({0})[0]
        capacity = capacity_for_rtf(rtf)
        first.capacity = capacity
        # Benchmark every added worker before deciding whether another is needed:
        # affinity sets and GPU contention can make profiles differ.
        while sum(worker.capacity for worker in self._workers) < len(self._config.streams):
            worker_id = len(self._workers)
            if worker_id >= self._config.workers.max_workers:
                self.stop()
                raise CapacityError(
                    f"{len(self._config.streams)} streams exceed the measured capacity of "
                    f"workers.max_workers = {self._config.workers.max_workers}"
                )
            worker = self._start_worker(worker_id)
            self._workers.append(worker)
            worker.capacity = capacity_for_rtf(self._await_benchmark({worker_id})[worker_id])
            if worker.capacity < 1:
                self.stop()
                raise CapacityError(f"worker {worker_id} cannot sustain a live stream")
        worker_count = len(self._workers)
        self._assign_streams()
        print(
            f"ready: {len(self._config.streams)} streams on {worker_count} worker(s); "
            f"benchmark RTF={rtf:.3f}, initial capacity={capacity}/worker",
            file=sys.stderr,
        )

    def _assign_streams(self) -> None:
        slots = [worker for worker in self._workers for _ in range(worker.capacity)]
        if len(slots) < len(self._config.streams):
            raise CapacityError("benchmark capacity changed and no longer covers every configured stream")
        for stream, worker in zip(self._config.streams, slots):
            self._states[stream.id] = StreamState(worker=worker)

    async def run(self) -> None:
        producers: list[asyncio.Task] = []
        try:
            self.start()
            producers = [asyncio.create_task(self._produce(stream.id)) for stream in self._config.streams]
            while True:
                self._drain_results()
                self._dispatch_pending()
                await asyncio.sleep(0.02)
        finally:
            self._stopping = True
            for producer in producers:
                producer.cancel()
            await asyncio.gather(*producers, return_exceptions=True)
            self.stop()

    async def _produce(self, stream_id: str) -> None:
        stream = next(stream for stream in self._config.streams if stream.id == stream_id)
        source = FfmpegAudioSource(stream, self._config.audio)
        async for window in source.windows():
            state = self._states[stream_id]
            # Exactly one queued window per stream: input never accumulates behind ASR.
            state.pending = window

    def _dispatch_pending(self) -> None:
        for stream_id, state in self._states.items():
            if state.inflight_sequence is not None or state.pending is None:
                continue
            state.next_sequence += 1
            window = state.pending
            state.pending = None
            state.inflight_sequence = state.next_sequence
            state.worker.input_queue.put(WorkItem(
                stream_id=stream_id,
                sequence=state.next_sequence,
                start_seconds=window.start_seconds,
                end_seconds=window.end_seconds,
                samples=window.samples,
            ))

    def _drain_results(self) -> None:
        while True:
            try:
                result = self._results.get_nowait()
            except Empty:
                return
            if not isinstance(result, WorkResult):
                continue
            state = self._states.get(result.stream_id)
            if state is None or state.inflight_sequence != result.sequence:
                continue
            state.inflight_sequence = None
            if result.error:
                print(f"[{result.stream_id}] transcription failed: {result.error}", file=sys.stderr)
                continue
            overrun = result.processing_seconds > result.audio_seconds
            self._sink.emit(TranscriptEvent(
                stream_id=result.stream_id,
                language=result.language,
                segments=result.segments,
                processing_seconds=result.processing_seconds,
                audio_seconds=result.audio_seconds,
                overrun=overrun,
            ))
            if overrun:
                # Result text is preserved, then stale captured audio is abandoned.
                state.pending = None
                print(f"[{result.stream_id}] RTF exceeded 1; dropped queued audio and resumed live", file=sys.stderr)

    def stop(self) -> None:
        for worker in self._workers:
            if worker.process.is_alive():
                worker.input_queue.put(None)
        for worker in self._workers:
            worker.process.join(timeout=10)
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(timeout=5)
        self._workers.clear()
