from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from queue import Empty
import sys
from urllib.parse import parse_qs, urlparse
from .config import ConfigurationError
from .poc import PocService, load_poc_config, rewrite_playlist


STATIC = Path(__file__).with_name("web")


class Handler(BaseHTTPRequestHandler):
    service: PocService

    def log_message(self, format: str, *args: object) -> None:
        print(f"[poc] {format % args}", file=sys.stderr)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                return self._file(STATIC / "index.html", "text/html; charset=utf-8")
            if parsed.path == "/hls.min.js":
                return self._file(STATIC / "hls.min.js", "application/javascript")
            if parsed.path == "/api/channels":
                channels = self.service.channels()
                return self._json([{
                    "id": channel.id, "name": channel.name, "video_codec": channel.video_codec,
                    "height": channel.height,
                } for channel in channels.values()])
            if parsed.path == "/api/captions":
                return self._captions(query)
            if parsed.path == "/media/playlist":
                return self._playlist(query)
            if parsed.path == "/media/segment":
                return self._segment(query)
            self.send_error(404)
        except (ValueError, RuntimeError) as error:
            self._json({"error": str(error)}, 400)
        except Exception as error:
            print(f"POC request failed: {error}", file=sys.stderr)
            self._json({"error": "upstream IPTV request failed"}, 502)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/select":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 1024:
                raise ValueError("invalid request body")
            channel_id = json.loads(self.rfile.read(length)).get("channel_id")
            if not isinstance(channel_id, str):
                raise ValueError("channel_id is required")
            playback = self.service.select(channel_id)
            self._json({"ok": True, "channel_id": channel_id, "mode": playback.mode})
        except (ValueError, RuntimeError) as error:
            self._json({"error": str(error)}, 400)
        except Exception as error:
            print(f"POC selection failed: {error}", file=sys.stderr)
            self._json({"error": "could not select channel"}, 502)

    def _playlist(self, query: dict[str, list[str]]) -> None:
        channel_id = self._one(query, "channel_id")
        playback = self.service.playback(channel_id)
        playlist = playback.manifest_path.read_text()
        self._bytes(rewrite_playlist(playlist, channel_id).encode(), "application/vnd.apple.mpegurl", 200)

    def _segment(self, query: dict[str, list[str]]) -> None:
        playback = self.service.playback(self._one(query, "channel_id"))
        data, content_type = self.service.read_media(playback, self._one(query, "path"))
        self._bytes(data, content_type, 200)

    def _captions(self, query: dict[str, list[str]]) -> None:
        channel_id = self._one(query, "channel_id")
        if self.service.transcriber.current_channel_id() != channel_id:
            raise ValueError("channel is not selected")
        subscriber = self.service.captions.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                try:
                    event = subscriber.get(timeout=15)
                    self.wfile.write(f"data: {event}\n\n".encode())
                except Empty:
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.service.captions.unsubscribe(subscriber)

    @staticmethod
    def _one(query: dict[str, list[str]], key: str) -> str:
        values = query.get(key)
        if not values or len(values) != 1 or not values[0]:
            raise ValueError(f"{key} is required")
        return values[0]

    def _file(self, path: Path, content_type: str) -> None:
        self._bytes(path.read_bytes(), content_type, 200)

    def _json(self, value: object, status: int = 200) -> None:
        self._bytes(json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

    def _bytes(self, data: bytes, content_type: str, status: int) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local IPTV ASR proof-of-concept player")
    parser.add_argument("--config", required=True, help="path to the POC TOML configuration")
    arguments = parser.parse_args()
    try:
        config = load_poc_config(arguments.config)
        service = PocService(config)
    except (ConfigurationError, RuntimeError) as error:
        print(f"POC startup error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    Handler.service = service
    server = ThreadingHTTPServer((config.bind_host, config.port), Handler)
    print(f"POC player: http://{config.bind_host}:{config.port}/", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.stop()


if __name__ == "__main__":
    main()
