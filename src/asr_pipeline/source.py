from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import sys
from typing import Callable

import numpy as np

from .config import AudioConfig, StreamConfig
from .contracts import AudioWindow


class FfmpegAudioSource:
    """Reconnectable live-source adapter that emits normalized fixed windows."""

    def __init__(self, stream: StreamConfig, config: AudioConfig, stop_requested: Callable[[], bool] | None = None) -> None:
        self._stream = stream
        self._config = config
        self._stop_requested = stop_requested or (lambda: False)

    def _command(self) -> list[str]:
        return [
            self._config.ffmpeg_path,
            "-nostdin", "-hide_banner", "-loglevel", "error",
            "-fflags", "+genpts+discardcorrupt",
            "-i", self._stream.url,
            "-map", "0:a:0?", "-ac", "1", "-ar", str(self._config.sample_rate),
            "-f", "s16le", "pipe:1",
        ]

    async def windows(self) -> AsyncIterator[AudioWindow]:
        bytes_per_window = int(self._config.sample_rate * self._config.window_seconds) * 2
        if bytes_per_window < 2:
            raise RuntimeError("audio window is too small")
        buffer = bytearray()
        stream_time = 0.0
        retry_delay = self._config.reconnect_initial_seconds
        while not self._stop_requested():
            process: asyncio.subprocess.Process | None = None
            try:
                process = await asyncio.create_subprocess_exec(
                    *self._command(), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
                )
                assert process.stdout is not None
                while data := await process.stdout.read(32_768):
                    if self._stop_requested():
                        return
                    buffer.extend(data)
                    while len(buffer) >= bytes_per_window:
                        raw = bytes(buffer[:bytes_per_window])
                        del buffer[:bytes_per_window]
                        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                        end_time = stream_time + len(samples) / self._config.sample_rate
                        yield AudioWindow(self._stream.id, stream_time, end_time, samples)
                        stream_time = end_time
                        retry_delay = self._config.reconnect_initial_seconds
                return_code = await process.wait()
                print(f"[{self._stream.id}] FFmpeg exited ({return_code}); reconnecting", file=sys.stderr)
            except FileNotFoundError as error:
                raise RuntimeError(f"FFmpeg executable not found: {self._config.ffmpeg_path}") from error
            except asyncio.CancelledError:
                raise
            except Exception as error:
                print(f"[{self._stream.id}] source error: {error}; reconnecting", file=sys.stderr)
            finally:
                if process is not None and process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, self._config.reconnect_max_seconds)
