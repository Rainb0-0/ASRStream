from __future__ import annotations

from dataclasses import dataclass, field
import json
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
from typing import Any, BinaryIO
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

from .config import ConfigurationError, PipelineConfig, load_config
from .contracts import AudioWindow
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
    stop_requested: threading.Event = field(default_factory=threading.Event)
    pending: AudioWindow | None = None
    inflight_sequence: int | None = None
    next_sequence: int = 0


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


class CaptionHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: set[Queue[str]] = set()

    def subscribe(self) -> Queue[str]:
        subscriber: Queue[str] = Queue(maxsize=16)
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: Queue[str]) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    def publish(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event, ensure_ascii=False)
        with self._lock:
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(payload)
            except Exception:
                # A slow browser does not get to delay live transcription.
                pass


class PocTranscriber:
    """One persistent local model worker with replaceable live-source sessions."""

    def __init__(self, config: PocConfig, captions: CaptionHub) -> None:
        self._config = config
        self._captions = captions
        self._context = mp.get_context("spawn")
        self._input: mp.Queue = self._context.Queue()
        self._output: mp.Queue = self._context.Queue()
        self._process = self._context.Process(
            target=worker_main,
            args=(0, WorkerSettings(config.pipeline.model, None, None), self._input, self._output),
            daemon=True,
        )
        self._lock = threading.Lock()
        self._session: Session | None = None
        self._generation = 0
        self._stopping = threading.Event()
        configure_cuda_libraries(config.pipeline.model)
        self._process.start()
        self._await_ready()
        self._results_thread = threading.Thread(target=self._drain_results, name="asr-results", daemon=True)
        self._results_thread.start()

    def _await_ready(self) -> None:
        try:
            message = self._output.get(timeout=180)
        except Empty as error:
            self.stop()
            raise RuntimeError("timed out loading local Faster-Whisper model") from error
        if isinstance(message, WorkerReady) and message.error is None:
            return
        self.stop()
        if isinstance(message, BenchmarkResult):
            raise RuntimeError(f"model worker failed: {message.error}")
        raise RuntimeError("model worker did not become ready")

    def select(self, channel: Channel, audio: BinaryIO) -> None:
        """Transcribe PCM emitted by the same FFmpeg ingest as playback."""
        with self._lock:
            if self._session is not None:
                self._session.stop_requested.set()
            self._generation += 1
            session = Session(self._generation, channel, f"poc-{self._generation}")
            self._session = session
        thread = threading.Thread(
            target=self._consume_source,
            args=(session, audio),
            name=f"audio-{channel.id}",
            daemon=True,
        )
        thread.start()

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
            if self._session is not session or session.stop_requested.is_set():
                return
            if session.inflight_sequence is not None:
                session.pending = window
                return
            self._dispatch_locked(session, window)

    def _dispatch_locked(self, session: Session, window: AudioWindow) -> None:
        session.next_sequence += 1
        session.inflight_sequence = session.next_sequence
        self._input.put(WorkItem(
            stream_id=session.internal_stream_id, sequence=session.next_sequence,
            start_seconds=window.start_seconds, end_seconds=window.end_seconds, samples=window.samples,
        ))

    def _drain_results(self) -> None:
        while not self._stopping.is_set():
            try:
                result = self._output.get(timeout=0.2)
            except Empty:
                continue
            if not isinstance(result, WorkResult):
                continue
            with self._lock:
                session = self._session
                if session is None or result.stream_id != session.internal_stream_id or result.sequence != session.inflight_sequence:
                    continue
                session.inflight_sequence = None
                if result.error:
                    print(f"[{session.channel.id}] transcription failed: {result.error}", file=sys.stderr)
                else:
                    for segment in result.segments:
                        self._captions.publish({
                            "channel_id": session.channel.id, "language": result.language,
                            "start_seconds": segment.start_seconds, "end_seconds": segment.end_seconds,
                            "text": segment.text,
                            "rtf": result.processing_seconds / result.audio_seconds if result.audio_seconds else None,
                        })
                    if result.processing_seconds > result.audio_seconds:
                        session.pending = None
                        print(f"[{session.channel.id}] RTF exceeded 1; dropped queued POC audio", file=sys.stderr)
                if session.pending is not None:
                    pending, session.pending = session.pending, None
                    self._dispatch_locked(session, pending)

    def current_channel_id(self) -> str | None:
        with self._lock:
            return self._session.channel.id if self._session else None

    def stop(self) -> None:
        self._stopping.set()
        with self._lock:
            if self._session is not None:
                self._session.stop_requested.set()
        if self._process.is_alive():
            self._input.put(None)
            self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)


class LocalCmafPackager:
    """Dedicated local playback adapter; it never changes the shared IPTV remuxer."""

    def __init__(self, config: PocConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._generation = 0
        shutil.rmtree(self._config.media_directory, ignore_errors=True)
        self._config.media_directory.mkdir(parents=True, exist_ok=True)

    def start(self, channel: Channel) -> PackagedMedia:
        with self._lock:
            self._stop_locked()
            # Keep old media files until shutdown: an HLS.js instance can still
            # request its previous playlist while the new selection starts.
            self._generation += 1
            output_directory = self._config.media_directory / str(self._generation)
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
        return manifest.read_text().count("#EXTINF:") >= 6

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
    def __init__(self, config: PocConfig) -> None:
        self.config = config
        self.iptv = IptvClient(config)
        self.captions = CaptionHub()
        self.transcriber = PocTranscriber(config, self.captions)
        self.packager = LocalCmafPackager(config)
        self._selection_lock = threading.Lock()
        self._playback_lock = threading.Lock()
        self._playback: Playback | None = None

    def channels(self) -> dict[str, Channel]:
        return self.iptv.channels()

    def select(self, channel_id: str) -> Playback:
        channel = self.channels().get(channel_id)
        if channel is None:
            raise ValueError("unknown or inactive channel")
        with self._selection_lock:
            media = self.packager.start(channel)
            # FFmpeg writes playback media and PCM from one ingest. Start
            # draining PCM before waiting for the player buffer so its pipe
            # cannot stall the CMAF packager.
            self.transcriber.select(channel, media.audio)
            self.packager.await_startup_buffer(media.manifest_path)
            playback = Playback(channel, media.manifest_path, "local_cmaf")
            with self._playback_lock:
                self._playback = playback
            return playback

    def playback(self, channel_id: str) -> Playback:
        with self._playback_lock:
            if self._playback is None or self._playback.channel.id != channel_id:
                raise ValueError("channel is not selected")
            return self._playback

    @staticmethod
    def read_media(playback: Playback, path: str) -> tuple[bytes, str]:
        candidate = (playback.manifest_path.parent / path).resolve()
        if playback.manifest_path.parent.resolve() not in candidate.parents:
            raise ValueError("invalid media path")
        content_type = "video/mp4" if candidate.suffix in {".m4s", ".mp4"} else "application/octet-stream"
        return candidate.read_bytes(), content_type

    def stop(self) -> None:
        self.packager.stop()
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
