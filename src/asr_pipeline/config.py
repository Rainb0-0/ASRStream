from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


class ConfigurationError(ValueError):
    """Raised when the external service configuration is invalid."""


@dataclass(frozen=True)
class ModelConfig:
    model_path: Path
    device: str
    device_index: int
    compute_type: str
    cpu_threads: int
    beam_size: int
    vad_filter: bool


@dataclass(frozen=True)
class WorkerConfig:
    max_workers: int
    core_sets: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class AudioConfig:
    ffmpeg_path: str
    sample_rate: int
    window_seconds: float
    reconnect_initial_seconds: float
    reconnect_max_seconds: float


@dataclass(frozen=True)
class StreamConfig:
    id: str
    url: str


@dataclass(frozen=True)
class PipelineConfig:
    model: ModelConfig
    workers: WorkerConfig
    audio: AudioConfig
    benchmark_audio: Path
    streams: tuple[StreamConfig, ...]


def _mapping(document: dict, key: str) -> dict:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f"[{key}] section is required")
    return value


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigurationError(f"{field} must be a positive integer")
    return value


def _positive_float(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{field} must be a positive number")
    return float(value)


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).expanduser()
    try:
        with config_path.open("rb") as file:
            document = tomllib.load(file)
    except FileNotFoundError as error:
        raise ConfigurationError(f"configuration file not found: {config_path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigurationError(f"invalid TOML in {config_path}: {error}") from error

    model_section = _mapping(document, "model")
    device = model_section.get("device")
    if device not in {"cpu", "cuda"}:
        raise ConfigurationError("model.device must be 'cpu' or 'cuda'")
    device_index = model_section.get("device_index", 0)
    if not isinstance(device_index, int) or isinstance(device_index, bool) or device_index < 0:
        raise ConfigurationError("model.device_index must be a non-negative integer")
    model_path_value = model_section.get("model_path")
    if not isinstance(model_path_value, str) or not model_path_value:
        raise ConfigurationError("model.model_path is required")
    model_path = Path(model_path_value).expanduser()
    if not model_path.is_absolute():
        model_path = config_path.parent / model_path
    if not model_path.is_dir():
        raise ConfigurationError(f"model directory not found: {model_path}")
    model = ModelConfig(
        model_path=model_path,
        device=device,
        device_index=device_index,
        compute_type=str(model_section.get("compute_type", "int8")),
        cpu_threads=_positive_int(model_section.get("cpu_threads", 1), "model.cpu_threads"),
        beam_size=_positive_int(model_section.get("beam_size", 5), "model.beam_size"),
        vad_filter=model_section.get("vad_filter", True),
    )
    if not isinstance(model.vad_filter, bool):
        raise ConfigurationError("model.vad_filter must be true or false")

    worker_section = _mapping(document, "workers")
    raw_core_sets = worker_section.get("core_sets", [])
    if not isinstance(raw_core_sets, list):
        raise ConfigurationError("workers.core_sets must be an array of core arrays")
    core_sets: list[tuple[int, ...]] = []
    for index, core_set in enumerate(raw_core_sets):
        if not isinstance(core_set, list) or not core_set:
            raise ConfigurationError(f"workers.core_sets[{index}] must be a non-empty array")
        if any(not isinstance(core, int) or isinstance(core, bool) or core < 0 for core in core_set):
            raise ConfigurationError(f"workers.core_sets[{index}] contains an invalid CPU index")
        core_sets.append(tuple(core_set))
    workers = WorkerConfig(
        max_workers=_positive_int(worker_section.get("max_workers"), "workers.max_workers"),
        core_sets=tuple(core_sets),
    )
    if workers.core_sets and len(workers.core_sets) < workers.max_workers:
        raise ConfigurationError("workers.core_sets must define one set for every possible worker")

    benchmark_section = _mapping(document, "benchmark")
    benchmark_value = benchmark_section.get("audio_path")
    if not isinstance(benchmark_value, str) or not benchmark_value:
        raise ConfigurationError("benchmark.audio_path is required")
    benchmark_audio = Path(benchmark_value).expanduser()
    if not benchmark_audio.is_file():
        raise ConfigurationError(f"benchmark audio file not found: {benchmark_audio}")

    audio_section = _mapping(document, "audio")
    audio = AudioConfig(
        ffmpeg_path=str(audio_section.get("ffmpeg_path", "ffmpeg")),
        sample_rate=_positive_int(audio_section.get("sample_rate", 16000), "audio.sample_rate"),
        window_seconds=_positive_float(audio_section.get("window_seconds", 5), "audio.window_seconds"),
        reconnect_initial_seconds=_positive_float(audio_section.get("reconnect_initial_seconds", 1), "audio.reconnect_initial_seconds"),
        reconnect_max_seconds=_positive_float(audio_section.get("reconnect_max_seconds", 30), "audio.reconnect_max_seconds"),
    )
    if audio.sample_rate != 16000:
        raise ConfigurationError("audio.sample_rate must be 16000 for Faster-Whisper input")
    if audio.reconnect_initial_seconds > audio.reconnect_max_seconds:
        raise ConfigurationError("audio.reconnect_initial_seconds must not exceed reconnect_max_seconds")

    sink_section = _mapping(document, "sink")
    if sink_section.get("type") != "stdout_ndjson":
        raise ConfigurationError("only sink.type = 'stdout_ndjson' is currently supported")

    raw_streams = document.get("streams")
    if not isinstance(raw_streams, list) or not raw_streams:
        raise ConfigurationError("at least one [[streams]] entry is required")
    streams: list[StreamConfig] = []
    stream_ids: set[str] = set()
    for index, entry in enumerate(raw_streams):
        if not isinstance(entry, dict):
            raise ConfigurationError(f"streams[{index}] must be a table")
        stream_id, url = entry.get("id"), entry.get("url")
        if not isinstance(stream_id, str) or not stream_id:
            raise ConfigurationError(f"streams[{index}].id is required")
        if stream_id in stream_ids:
            raise ConfigurationError(f"duplicate stream id: {stream_id}")
        if not isinstance(url, str) or not url.startswith(("http://", "https://", "udp://", "rtsp://")):
            raise ConfigurationError(f"streams[{index}].url must use http(s), udp, or rtsp")
        stream_ids.add(stream_id)
        streams.append(StreamConfig(id=stream_id, url=url))

    return PipelineConfig(model=model, workers=workers, audio=audio, benchmark_audio=benchmark_audio, streams=tuple(streams))
