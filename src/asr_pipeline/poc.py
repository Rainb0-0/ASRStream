from __future__ import annotations

from dataclasses import dataclass, field
import json
import hashlib
import multiprocessing as mp
from pathlib import Path
from queue import Empty, Queue
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from typing import Any, BinaryIO, Protocol
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

from .config import ConfigurationError, PipelineConfig, load_config
from .contracts import AudioWindow
from .pipeline import CapacityError, capacity_for_rtf
from .worker import BenchmarkResult, WorkItem, WorkResult, WorkerReady, WorkerSettings, configure_cuda_libraries, worker_main


@dataclass(frozen=True)
class PocConfig:
    pipeline: PipelineConfig
    bind_host: str
    port: int
    channel_api: str
    credentials_env: Path
    username_variable: str
    password_variable: str
    media_directory: Path


@dataclass(frozen=True)
class PublicPocConfig:
    pipeline: PipelineConfig
    bind_host: str
    port: int
    playlist_url: str
    playlist_cache_path: Path
    media_directory: Path


@dataclass(frozen=True)
class Channel:
    id: str
    name: str
    hls_url: str
    video_codec: str | None
    height: int | None


@dataclass(frozen=True)
class Playback:
    channel: Channel
    manifest_path: Path
    mode: str


@dataclass(frozen=True)
class PackagedMedia:
    manifest_path: Path
    audio: BinaryIO


@dataclass
class Session:
    generation: int
    channel: Channel
    internal_stream_id: str
    worker: "PocWorker"
    stop_requested: threading.Event = field(default_factory=threading.Event)
    pending: AudioWindow | None = None
    inflight_sequence: int | None = None
    next_sequence: int = 0


@dataclass
class PocWorker:
    worker_id: int
    process: mp.Process
    input_queue: mp.Queue
    capacity: int
    stream_ids: set[str] = field(default_factory=set)


def load_poc_config(path: str | Path) -> PocConfig:
    pipeline = load_config(path)
    config_path = Path(path).expanduser()
    with config_path.open("rb") as file:
        document = tomllib.load(file)
    section = document.get("poc")
    if not isinstance(section, dict):
        raise ConfigurationError("[poc] section is required")

    def string(key: str) -> str:
        value = section.get(key)
        if not isinstance(value, str) or not value:
            raise ConfigurationError(f"poc.{key} is required")
        return value

    port = section.get("port")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ConfigurationError("poc.port must be a TCP port number")
    credentials_env = Path(string("credentials_env")).expanduser()
    if not credentials_env.is_absolute():
        credentials_env = config_path.parent / credentials_env
    if not credentials_env.is_file():
        raise ConfigurationError(f"POC credentials file not found: {credentials_env}")
    media_directory = Path(string("media_directory")).expanduser()
    if not media_directory.is_absolute():
        media_directory = config_path.parent / media_directory
    return PocConfig(
        pipeline=pipeline,
        bind_host=string("bind_host"),
        port=port,
        channel_api=string("channel_api"),
        credentials_env=credentials_env,
        username_variable=string("username_variable"),
        password_variable=string("password_variable"),
        media_directory=media_directory,
    )


def load_public_poc_config(path: str | Path) -> PublicPocConfig:
    pipeline = load_config(path)
    config_path = Path(path).expanduser()
    with config_path.open("rb") as file:
        document = tomllib.load(file)
    section = document.get("poc_public")
    if not isinstance(section, dict):
        raise ConfigurationError("[poc_public] section is required")

    def string(key: str) -> str:
        value = section.get(key)
        if not isinstance(value, str) or not value:
            raise ConfigurationError(f"poc_public.{key} is required")
        return value

    port = section.get("port")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ConfigurationError("poc_public.port must be a TCP port number")
    playlist_url = string("playlist_url")
    if not playlist_url.startswith(("http://", "https://")):
        raise ConfigurationError("poc_public.playlist_url must be an HTTP(S) URL")
    playlist_cache_path = Path(string("playlist_cache_path")).expanduser()
    if not playlist_cache_path.is_absolute():
        playlist_cache_path = config_path.parent / playlist_cache_path
    media_directory = Path(string("media_directory")).expanduser()
    if not media_directory.is_absolute():
        media_directory = config_path.parent / media_directory
    return PublicPocConfig(
        pipeline=pipeline,
        bind_host=string("bind_host"),
        port=port,
        playlist_url=playlist_url,
        playlist_cache_path=playlist_cache_path,
        media_directory=media_directory,
    )


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or not key.replace("_", "").isalnum():
            continue
        values[key] = value.strip().strip("\"'")
    return values


class IptvClient:
    def __init__(self, config: PocConfig) -> None:
        self._config = config

    def channels(self) -> dict[str, Channel]:
        credentials = _read_env_file(self._config.credentials_env)
        try:
            username = credentials[self._config.username_variable]
            password = credentials[self._config.password_variable]
        except KeyError as error:
            raise RuntimeError(f"credentials file is missing {error.args[0]}") from error
        query = urlencode({"channels": "", "type": "iptv", "u": username, "p": password, "debugp": "0"})
        with urlopen(f"{self._config.channel_api}?{query}", timeout=20) as response:
            records = json.load(response)
        if not isinstance(records, list):
            raise RuntimeError("IPTV channel API did not return a channel list")
        channels: dict[str, Channel] = {}
        for record in records:
            if not isinstance(record, dict) or not (record.get("active") and record.get("hls")):
                continue
            channel_id, name, url = record.get("_id"), record.get("name"), record.get("url_hls")
            if channel_id is None or not isinstance(name, str) or not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            channels[str(channel_id)] = Channel(
                id=str(channel_id), name=name, hls_url=url,
                video_codec=record.get("video_codec") if isinstance(record.get("video_codec"), str) else None,
                height=record.get("height") if isinstance(record.get("height"), int) else None,
            )
        return channels


class ChannelCatalog(Protocol):
    def channels(self) -> dict[str, Channel]: ...


def parse_m3u_channels(playlist: str) -> dict[str, Channel]:
    """Parse an extended M3U catalog, retaining direct FFmpeg-compatible URLs."""
    channels: dict[str, Channel] = {}
    name: str | None = None
    for raw_line in playlist.splitlines():
        line = raw_line.strip()
        if line.startswith("#EXTINF:"):
            _, _, title = line.partition(",")
            name = title.strip() or "Unnamed channel"
            continue
        if not line or line.startswith("#"):
            continue
        if name is not None and line.startswith(("http://", "https://", "rtsp://", "udp://")):
            channel_id = hashlib.sha256(line.encode()).hexdigest()[:20]
            channels[channel_id] = Channel(channel_id, name, line, None, None)
        name = None
    return channels


class PublicPlaylistClient:
    def __init__(self, config: PublicPocConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._channels: dict[str, Channel] | None = None

    def channels(self) -> dict[str, Channel]:
        with self._lock:
            if self._channels is None:
                cache = self._config.playlist_cache_path
                if cache.is_file():
                    playlist = cache.read_text(encoding="utf-8-sig", errors="replace")
                else:
                    with urlopen(self._config.playlist_url, timeout=30) as response:
                        payload = response.read()
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    temporary = cache.with_name(cache.name + ".tmp")
                    temporary.write_bytes(payload)
                    temporary.replace(cache)
                    playlist = payload.decode("utf-8-sig", errors="replace")
                self._channels = parse_m3u_channels(playlist)
                if not self._channels:
                    raise RuntimeError("public IPTV playlist contains no playable channels")
            return self._channels


class CaptionHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[Queue[str], str] = {}

    def subscribe(self, channel_id: str) -> Queue[str]:
        subscriber: Queue[str] = Queue(maxsize=16)
        with self._lock:
            self._subscribers[subscriber] = channel_id
        return subscriber

    def unsubscribe(self, subscriber: Queue[str]) -> None:
        with self._lock:
            self._subscribers.pop(subscriber, None)

    def publish(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event, ensure_ascii=False)
        with self._lock:
            subscribers = tuple(
                subscriber for subscriber, channel_id in self._subscribers.items()
                if channel_id == event["channel_id"]
            )
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(payload)
            except Exception:
                # A slow browser does not get to delay live transcription.
                pass


class PocTranscriber:
    """Capacity-aware local workers with one independently scheduled session per channel."""

    def __init__(self, config: PocConfig | PublicPocConfig, captions: CaptionHub) -> None:
        self._config = config
        self._captions = captions
        self._context = mp.get_context("spawn")
        self._output: mp.Queue = self._context.Queue()
        self._benchmark_results: Queue[BenchmarkResult] = Queue()
        self._lock = threading.Lock()
        self._worker_lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._workers: list[PocWorker] = []
        self._generation = 0
        self._stopping = threading.Event()

        first = self._start_worker(0)
        self._workers.append(first)
        first.capacity = self._await_benchmark(0)
        self._results_thread = threading.Thread(target=self._drain_results, name="asr-results", daemon=True)
        self._results_thread.start()
        print(f"POC ASR ready: capacity={first.capacity} stream(s)/worker", file=sys.stderr)

    def _start_worker(self, worker_id: int) -> PocWorker:
        configure_cuda_libraries(self._config.pipeline.model)
        core_sets = self._config.pipeline.workers.core_sets
        core_set = core_sets[worker_id] if core_sets else None
        input_queue: mp.Queue = self._context.Queue()
        settings = WorkerSettings(
            self._config.pipeline.model,
            str(self._config.pipeline.benchmark_audio),
            core_set,
        )
        process = self._context.Process(
            target=worker_main,
            args=(worker_id, settings, input_queue, self._output),
            daemon=True,
        )
        process.start()
        return PocWorker(worker_id, process, input_queue, capacity=0)

    def _await_benchmark(self, worker_id: int) -> int:
        try:
            while True:
                message = self._output.get(timeout=180) if not self._results_thread_alive() else self._benchmark_results.get(timeout=180)
                if isinstance(message, BenchmarkResult) and message.worker_id == worker_id:
                    if message.error:
                        raise CapacityError(f"worker {worker_id} benchmark failed: {message.error}")
                    assert message.rtf is not None
                    return capacity_for_rtf(message.rtf)
        except Empty as error:
            raise CapacityError("timed out waiting for Faster-Whisper startup benchmark") from error

    def _results_thread_alive(self) -> bool:
        return hasattr(self, "_results_thread") and self._results_thread.is_alive()

    def _allocate_worker(self) -> PocWorker:
        with self._worker_lock:
            candidates = [worker for worker in self._workers if len(worker.stream_ids) < worker.capacity]
            if candidates:
                return min(candidates, key=lambda worker: len(worker.stream_ids))
            worker_id = len(self._workers)
            if worker_id >= self._config.pipeline.workers.max_workers:
                raise CapacityError(
                    f"active channels exceed the measured capacity of workers.max_workers = "
                    f"{self._config.pipeline.workers.max_workers}"
                )
            worker = self._start_worker(worker_id)
            self._workers.append(worker)
            try:
                worker.capacity = self._await_benchmark(worker_id)
            except Exception:
                self._workers.remove(worker)
                self._stop_worker(worker)
                raise
            print(f"POC ASR added worker {worker_id}: capacity={worker.capacity} stream(s)", file=sys.stderr)
            return worker

    def select(self, channel: Channel, audio: BinaryIO) -> None:
        """Transcribe PCM emitted by the selected channel's dedicated FFmpeg ingest."""
        with self._lock:
            if channel.id in self._sessions:
                return
        worker = self._allocate_worker()
        with self._lock:
            if channel.id in self._sessions:
                return
            self._generation += 1
            session = Session(self._generation, channel, f"poc-{self._generation}", worker)
            self._sessions[session.internal_stream_id] = session
            worker.stream_ids.add(session.internal_stream_id)
        threading.Thread(
            target=self._consume_source,
            args=(session, audio),
            name=f"audio-{channel.id}",
            daemon=True,
        ).start()

    def _consume_source(self, session: Session, audio: BinaryIO) -> None:
        bytes_per_window = int(self._config.pipeline.audio.sample_rate * self._config.pipeline.audio.window_seconds) * 2
        buffer = bytearray()
        stream_time = 0.0
        try:
            while not session.stop_requested.is_set():
                data = audio.read(32_768)
                if not data:
                    return
                buffer.extend(data)
                while len(buffer) >= bytes_per_window:
                    raw = bytes(buffer[:bytes_per_window])
                    del buffer[:bytes_per_window]
                    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                    end_time = stream_time + len(samples) / self._config.pipeline.audio.sample_rate
                    window = AudioWindow(session.internal_stream_id, stream_time, end_time, samples)
                    stream_time = end_time
                    self._submit(session, window)
        except Exception as error:
            if not session.stop_requested.is_set():
                print(f"[{session.channel.id}] POC audio source stopped: {error}", file=sys.stderr)

    def _submit(self, session: Session, window: AudioWindow) -> None:
        with self._lock:
            if self._sessions.get(session.internal_stream_id) is not session or session.stop_requested.is_set():
                return
            if session.inflight_sequence is not None:
                session.pending = window
                return
            self._dispatch_locked(session, window)

    def _dispatch_locked(self, session: Session, window: AudioWindow) -> None:
        session.next_sequence += 1
        session.inflight_sequence = session.next_sequence
        session.worker.input_queue.put(WorkItem(
            stream_id=session.internal_stream_id, sequence=session.next_sequence,
            start_seconds=window.start_seconds, end_seconds=window.end_seconds, samples=window.samples,
        ))

    def _drain_results(self) -> None:
        while not self._stopping.is_set():
            try:
                result = self._output.get(timeout=0.2)
            except Empty:
                continue
            if isinstance(result, BenchmarkResult):
                self._benchmark_results.put(result)
                continue
            if not isinstance(result, WorkResult):
                continue
            with self._lock:
                session = self._sessions.get(result.stream_id)
                if session is None or result.sequence != session.inflight_sequence:
                    continue
                session.inflight_sequence = None
                if result.error:
                    print(f"[{session.channel.id}] transcription failed: {result.error}", file=sys.stderr)
                else:
                    rtf = result.processing_seconds / result.audio_seconds if result.audio_seconds else None
                    self._captions.publish({
                        "type": "metrics", "channel_id": session.channel.id, "rtf": rtf,
                    })
                    for segment in result.segments:
                        self._captions.publish({
                            "type": "caption", "channel_id": session.channel.id, "language": result.language,
                            "start_seconds": segment.start_seconds, "end_seconds": segment.end_seconds,
                            "text": segment.text,
                            "rtf": rtf,
                        })
                    if result.processing_seconds > result.audio_seconds:
                        session.pending = None
                        print(f"[{session.channel.id}] RTF exceeded 1; dropped queued POC audio", file=sys.stderr)
                if session.pending is not None:
                    pending, session.pending = session.pending, None
                    self._dispatch_locked(session, pending)

    def has_channel(self, channel_id: str) -> bool:
        with self._lock:
            return any(session.channel.id == channel_id for session in self._sessions.values())

    def stop(self) -> None:
        self._stopping.set()
        with self._lock:
            for session in self._sessions.values():
                session.stop_requested.set()
        for worker in self._workers:
            self._stop_worker(worker)
        self._workers.clear()

    @staticmethod
    def _stop_worker(worker: PocWorker) -> None:
        if worker.process.is_alive():
            worker.input_queue.put(None)
            worker.process.join(timeout=10)
        if worker.process.is_alive():
            worker.process.terminate()
            worker.process.join(timeout=5)


class LocalCmafPackager:
    """Dedicated local playback adapter; it never changes the shared IPTV remuxer."""

    def __init__(self, config: PocConfig | PublicPocConfig, media_directory: Path) -> None:
        self._config = config
        self._media_directory = media_directory
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._generation = 0
        self._media_directory.mkdir(parents=True, exist_ok=True)

    def start(self, channel: Channel) -> PackagedMedia:
        with self._lock:
            self._stop_locked()
            # Keep old media files until shutdown: an HLS.js instance can still
            # request its previous playlist while the new selection starts.
            self._generation += 1
            output_directory = self._media_directory / str(self._generation)
            output_directory.mkdir(parents=True, exist_ok=True)
            manifest = output_directory / "stream.m3u8"
            video_tag = "hvc1" if (channel.video_codec or "").lower() in {"hevc", "h265"} else "avc1"
            command = [
                self._config.pipeline.audio.ffmpeg_path,
                "-nostdin", "-hide_banner", "-loglevel", "warning",
                "-fflags", "+genpts+discardcorrupt", "-i", channel.hls_url,
                "-map", "0:v:0", "-map", "0:a:0?",
                "-c:v", "copy", "-tag:v", video_tag,
                "-c:a", "aac", "-profile:a", "aac_low", "-b:a", "128k", "-ar", "48000",
                "-f", "hls", "-hls_segment_type", "fmp4",
                "-hls_fmp4_init_filename", "init.mp4", "-hls_time", "2",
                "-hls_list_size", "16", "-hls_delete_threshold", "4",
                "-hls_flags", "delete_segments+independent_segments+program_date_time",
                "-hls_segment_filename", str(output_directory / "segment_%06d.m4s"),
                str(manifest),
                "-map", "0:a:0?", "-ac", "1", "-ar", str(self._config.pipeline.audio.sample_rate),
                "-f", "s16le", "pipe:1",
            ]
            self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if self._process.stdout is None:
                raise RuntimeError("could not create local ASR audio stream")
            return PackagedMedia(manifest, self._process.stdout)

    def await_startup_buffer(self, manifest: Path) -> None:
        with self._lock:
            if self._process is None:
                raise RuntimeError("local CMAF packager is not running")
            for _ in range(80):
                if self._has_startup_buffer(manifest):
                    return
                if self._process.poll() is not None:
                    error = self._process.stderr.read().decode(errors="replace") if self._process.stderr else ""
                    raise RuntimeError(f"local CMAF packager exited ({self._process.returncode}): {error[-500:]}")
                time.sleep(0.5)
            self._stop_locked()
            raise RuntimeError("timed out waiting for the local CMAF playlist")

    @staticmethod
    def _has_startup_buffer(manifest: Path) -> bool:
        if not manifest.is_file() or not (manifest.parent / "init.mp4").is_file():
            return False
        # A single fMP4 segment is enough for HLS.js to start. Requiring a
        # six-segment live buffer here wrongly rejects slower public streams.
        return "#EXTINF:" in manifest.read_text()

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self._process is None or self._process.poll() is not None:
            self._process = None
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        self._process = None


class PocService:
    def __init__(self, config: PocConfig | PublicPocConfig, catalog: ChannelCatalog) -> None:
        self.config = config
        self.catalog = catalog
        self.captions = CaptionHub()
        self.transcriber = PocTranscriber(config, self.captions)
        shutil.rmtree(self.config.media_directory, ignore_errors=True)
        self.config.media_directory.mkdir(parents=True, exist_ok=True)
        self._selection_lock = threading.Lock()
        self._playback_lock = threading.Lock()
        self._playbacks: dict[str, Playback] = {}
        self._packagers: dict[str, LocalCmafPackager] = {}

    def channels(self) -> dict[str, Channel]:
        return self.catalog.channels()

    def select(self, channel_id: str) -> Playback:
        channel = self.channels().get(channel_id)
        if channel is None:
            raise ValueError("unknown or inactive channel")
        with self._selection_lock:
            with self._playback_lock:
                existing = self._playbacks.get(channel_id)
            if existing is not None:
                return existing
            media_directory = self.config.media_directory / hashlib.sha256(channel_id.encode()).hexdigest()[:20]
            packager = LocalCmafPackager(self.config, media_directory)
            media = packager.start(channel)
            # FFmpeg writes playback media and PCM from one ingest. Start
            # draining PCM before waiting for the player buffer so its pipe
            # cannot stall the CMAF packager.
            try:
                self.transcriber.select(channel, media.audio)
                packager.await_startup_buffer(media.manifest_path)
            except Exception:
                packager.stop()
                raise
            playback = Playback(channel, media.manifest_path, "local_cmaf")
            with self._playback_lock:
                self._playbacks[channel_id] = playback
                self._packagers[channel_id] = packager
            return playback

    def playback(self, channel_id: str) -> Playback:
        with self._playback_lock:
            playback = self._playbacks.get(channel_id)
            if playback is None:
                raise ValueError("channel is not selected")
            return playback

    @staticmethod
    def read_media(playback: Playback, path: str) -> tuple[bytes, str]:
        candidate = (playback.manifest_path.parent / path).resolve()
        if playback.manifest_path.parent.resolve() not in candidate.parents:
            raise ValueError("invalid media path")
        content_type = "video/mp4" if candidate.suffix in {".m4s", ".mp4"} else "application/octet-stream"
        return candidate.read_bytes(), content_type

    def stop(self) -> None:
        with self._playback_lock:
            packagers = tuple(self._packagers.values())
            self._packagers.clear()
            self._playbacks.clear()
        for packager in packagers:
            packager.stop()
        self.transcriber.stop()


_URI_ATTRIBUTE = re.compile(r'URI="([^"]+)"')


def rewrite_playlist(playlist: str, channel_id: str) -> str:
    def proxy(uri: str) -> str:
        return "/media/segment?" + urlencode({"channel_id": channel_id, "path": uri})

    output: list[str] = []
    for line in playlist.splitlines():
        if line.startswith("#"):
            output.append(_URI_ATTRIBUTE.sub(lambda match: f'URI="{proxy(match.group(1))}"', line))
        elif line:
            output.append(proxy(line))
        else:
            output.append(line)
    return "\n".join(output) + "\n"
