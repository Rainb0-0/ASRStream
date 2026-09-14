from __future__ import annotations

import tempfile
import threading
import unittest
from unittest import mock
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import json
from queue import Empty

from asr_pipeline.config import ConfigurationError, load_config
from asr_pipeline.contracts import TranscriptEvent, TranscriptSegment
from asr_pipeline.pipeline import CapacityError, capacity_for_rtf, required_worker_count
from asr_pipeline.poc import CaptionHub, Channel, LocalCmafPackager, Playback, PocService, PocTranscriber, PublicPlaylistClient, PublicPocConfig, Session, parse_m3u_channels, rewrite_playlist
from asr_pipeline.sink import NdjsonStdoutSink


class ConfigurationTests(unittest.TestCase):
    def _config(self, directory: Path, extra: str = "") -> Path:
        benchmark = directory / "benchmark.wav"
        benchmark.write_bytes(b"placeholder")
        model = directory / "faster-whisper-large-v3"
        model.mkdir()
        config = directory / "config.toml"
        config.write_text(f'''
[model]
model_path = "{model}"
device = "cpu"
cpu_threads = 2

[workers]
max_workers = 2

[benchmark]
audio_path = "{benchmark}"

[audio]
window_seconds = 5

[sink]
type = "stdout_ndjson"

[[streams]]
id = "one"
url = "http://example.test/live.m3u8"
{extra}
''')
        return config

    def test_loads_valid_minimal_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(self._config(Path(directory)))
        self.assertEqual(config.model.device, "cpu")
        self.assertEqual(config.model.cpu_threads, 2)
        self.assertEqual(config.streams[0].id, "one")

    def test_rejects_duplicate_streams(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._config(Path(directory), '''
[[streams]]
id = "one"
url = "http://example.test/another.m3u8"
''')
            with self.assertRaisesRegex(ConfigurationError, "duplicate stream id"):
                load_config(config_path)

    def test_rejects_unknown_sink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._config(Path(directory))
            config_path.write_text(config_path.read_text().replace('stdout_ndjson', 'kafka'))
            with self.assertRaisesRegex(ConfigurationError, "only sink.type"):
                load_config(config_path)


class CapacityTests(unittest.TestCase):
    def test_conventional_rtf_capacity(self) -> None:
        self.assertEqual(capacity_for_rtf(0.40), 2)
        self.assertEqual(required_worker_count(5, 2), 3)

    def test_unsustainable_worker_is_rejected(self) -> None:
        with self.assertRaises(CapacityError):
            required_worker_count(1, capacity_for_rtf(1.0))


class OutputTests(unittest.TestCase):
    def test_ndjson_sink_emits_finalized_segments(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            NdjsonStdoutSink().emit(TranscriptEvent(
                stream_id="tv1",
                language="fa",
                segments=(TranscriptSegment(1.0, 2.0, "سلام"),),
                processing_seconds=0.3,
                audio_seconds=5.0,
                overrun=False,
            ))
        event = json.loads(output.getvalue())
        self.assertEqual(event["stream_id"], "tv1")
        self.assertEqual(event["segments"][0]["text"], "سلام")
        self.assertEqual(event["rtf"], 0.06)


class PocPlaylistTests(unittest.TestCase):
    def test_playlist_proxy_rewrites_segments_and_initialization_media(self) -> None:
        source = '#EXTM3U\n#EXT-X-MAP:URI="init.mp4"\nsegment001.m4s\n'
        rewritten = rewrite_playlist(source, "1001761")
        self.assertIn("channel_id=1001761", rewritten)
        self.assertIn("path=init.mp4", rewritten)
        self.assertIn("path=segment001.m4s", rewritten)

    def test_public_m3u_parser_keeps_supported_streams_with_stable_ids(self) -> None:
        playlist = """#EXTM3U
#EXTINF:-1 tvg-id=\"one\",News One
https://example.test/news.m3u8
#EXTINF:-1,Unsupported
file:///tmp/local.ts
"""
        channels = parse_m3u_channels(playlist)
        self.assertEqual(len(channels), 1)
        channel = next(iter(channels.values()))
        self.assertEqual(channel.name, "News One")
        self.assertEqual(channel.hls_url, "https://example.test/news.m3u8")

    def test_public_playlist_client_uses_its_local_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = load_config(ConfigurationTests()._config(root))
            cache = root / "iptv" / "index.m3u"
            cache.parent.mkdir()
            cache.write_text("#EXTM3U\n#EXTINF:-1,Cached\nhttps://example.test/cached.m3u8\n")
            config = PublicPocConfig(pipeline, "127.0.0.1", 8081, "https://invalid.test/index.m3u", cache, root / "media")
            channels = PublicPlaylistClient(config).channels()
            self.assertEqual(next(iter(channels.values())).name, "Cached")

    def test_public_master_resolves_to_live_variant_and_preserves_redirect_query(self) -> None:
        master = "https://edge.example/live/playlist.m3u8?isp=tci"
        with mock.patch("asr_pipeline.poc.urlopen") as open_url:
            master_response = mock.MagicMock()
            master_response.geturl.return_value = master
            master_response.read.return_value = (
                b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\n240p/index.m3u8\n"
            )
            variant_response = mock.MagicMock()
            variant_response.geturl.return_value = "https://live-edge.example/live/240p/index.m3u8?isp=tci"
            variant_response.read.return_value = b"#EXTM3U\n#EXTINF:2,\nsegment.ts\n"
            segment_response = mock.MagicMock()
            segment_response.read.return_value = b"x" * 188
            open_url.return_value.__enter__.side_effect = [master_response, variant_response, segment_response]
            resolved = PublicPlaylistClient.resolve(Channel("id", "IRIB1", master, None, None))
        self.assertEqual(resolved.hls_url, "https://live-edge.example/live/240p/index.m3u8?isp=tci")


class PocMultichannelTests(unittest.TestCase):
    def test_caption_hub_routes_each_caption_to_its_channel(self) -> None:
        captions = CaptionHub()
        first = captions.subscribe("first")
        second = captions.subscribe("second")
        captions.publish({"channel_id": "first", "text": "first caption"})
        self.assertIn("first caption", first.get_nowait())
        with self.assertRaises(Empty):
            second.get_nowait()

    def test_playback_lookup_keeps_multiple_selected_channels(self) -> None:
        service = object.__new__(PocService)
        service._playback_lock = threading.Lock()
        first = Playback(Channel("first", "First", "https://first.test/live", None, None), Path("/tmp/first.m3u8"), "local_cmaf")
        second = Playback(Channel("second", "Second", "https://second.test/live", None, None), Path("/tmp/second.m3u8"), "local_cmaf")
        service._playbacks = {"first": first, "second": second}
        self.assertIs(service.playback("first"), first)
        self.assertIs(service.playback("second"), second)

    def test_deselected_channel_is_removed_from_asr_sessions(self) -> None:
        transcriber = object.__new__(PocTranscriber)
        transcriber._lock = threading.Lock()
        worker = type("Worker", (), {"stream_ids": {"poc-1"}})()
        session = Session(
            1,
            Channel("first", "First", "https://first.test/live", None, None),
            "poc-1",
            worker,
        )
        transcriber._sessions = {session.internal_stream_id: session}

        transcriber.stop_channels({"first"})

        self.assertTrue(session.stop_requested.is_set())
        self.assertNotIn(session.internal_stream_id, transcriber._sessions)
        self.assertNotIn(session.internal_stream_id, worker.stream_ids)

    def test_prompt_uses_recent_transcript_history_for_its_channel(self) -> None:
        transcriber = object.__new__(PocTranscriber)
        worker = type("Worker", (), {"stream_ids": set()})()
        session = Session(1, Channel("one", "One", "https://one.test/live", None, None), "poc-1", worker)
        session.prompt_history.extend([(10.0, "old"), (190.0, "recent context")])

        self.assertEqual(transcriber._prompt_for(session, 200.0), "recent context")

    def test_cmaf_packager_waits_for_the_live_sync_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory)
            manifest = media / "stream.m3u8"
            (media / "init.mp4").write_bytes(b"init")
            manifest.write_text("#EXTM3U\n#EXTINF:6,\nfirst.m4s\n#EXTINF:6,\nsecond.m4s\n")
            self.assertTrue(LocalCmafPackager._has_startup_buffer(manifest))

    def test_cmaf_packager_reconnects_when_live_http_input_closes(self) -> None:
        command_source = Path(LocalCmafPackager.start.__code__.co_filename).read_text()
        self.assertIn('"-reconnect", "1"', command_source)
        self.assertIn('"-reconnect_streamed", "1"', command_source)
        self.assertIn('"-reconnect_on_network_error", "1"', command_source)
        self.assertIn('"-live_start_index", "-3"', command_source)
        self.assertIn('"-allowed_extensions", "ALL"', command_source)
        self.assertIn('"-allowed_segment_extensions", "ALL"', command_source)


if __name__ == "__main__":
    unittest.main()
