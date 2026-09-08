# IPTV ASR pipeline

This service transcribes one or more live IPTV audio streams with
Faster-Whisper `large-v3`.  It reads a TOML configuration, uses FFmpeg to
normalize each source to 16 kHz mono PCM, and emits finalized transcript
segments as NDJSON on standard output.

## Prerequisites

- Python 3.11 or later
- FFmpeg available as `ffmpeg` (or configured with `ffmpeg_path`)
- Faster-Whisper runtime dependencies. CUDA mode also needs a compatible
  CTranslate2/CUDA installation.
- A representative local WAV file for startup capacity measurement.

Install the package, copy `config.example.toml` to a deployment-specific path,
and set the absolute model path, stream URLs, and benchmark file path. The
pipeline never downloads a model: `model.model_path` must point to the copied
local Large-v3 directory.

```bash
python -m pip install .
asr-pipeline --config /etc/asr-pipeline/config.toml
```

Operational messages are written to stderr. Standard output is reserved for
one JSON transcript event per line, so it can be piped into another process.

The service determines capacity by measuring conventional RTF
(`transcription wall time / audio duration`) at startup. A worker is assigned
`floor(1 / RTF)` streams. If it cannot sustain one stream, or configured
resource limits cannot cover every stream, startup fails rather than silently
running behind live.

When a stream falls behind at runtime, the current completed segments are
emitted, its queued audio is discarded, and transcription continues at the
live edge.

## Local player proof of concept

The production ASR pipeline lives in `src/asr_pipeline/{pipeline,source,worker,sink}.py`.
The browser-only proof of concept is kept separately in
`src/asr_pipeline/{poc,poc_cli}.py` and `src/asr_pipeline/web/`.

With the company IPTV tunnel running, bootstrap the Python 3.12 environment,
install the POC dependencies, and launch the single-channel browser player
with:

```bash
./run-poc.sh
```

Pass a deployment-specific config as the first argument when needed:

```bash
./run-poc.sh /etc/asr-pipeline/poc.toml
```

The launcher installs the CUDA runtime wheels only when the selected config
uses `device = "cuda"`. It does not download the Large-v3 model; the config
must continue to point at an existing local model directory.

To run the already-installed entry point directly:

```bash
asr-poc --config config.example.toml
```

Open `http://127.0.0.1:8080/`. The service resolves channels through the
authenticated API, packages the selected channel locally as CMAF with AAC-LC
audio, and sends finalized Large-v3 segments to the subtitle overlay via SSE.
The subtitle switch changes only display; local transcription remains active
for the selected channel.

The checked-in example targets the local RTX 4060 (`device = "cuda"`,
`compute_type = "float16"`). CUDA must be visible to the process; restricted
sandbox environments intentionally do not expose the GPU even when the normal
desktop shell does.
