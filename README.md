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
core pipeline expects `model.model_path` to point to a local Large-v3
directory; the launchers below can download that model on first run.

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
uses `device = "cuda"`. On first run it downloads
`Systran/faster-whisper-large-v3` from Hugging Face into the configured
`model_path`; later runs reuse the local copy. This requires network access
and several GB of disk space.

To run the already-installed entry point directly:

```bash
asr-poc --config config.example.toml
```

## Public IPTV-org proof of concept

The public POC is separate from the company-tunnel POC. It downloads the
current [iptv-org public catalog](https://iptv-org.github.io/iptv/index.m3u),
shows its channels in the same player, and transcribes the selected stream
locally. It has no credential or tunnel dependency.

### Public POC requirements

- Windows 10/11 with Windows PowerShell 5.1 or PowerShell 7+
- Python 3.12 available as `py -3.12` or `python`
- FFmpeg available on `PATH` (or configured with `audio.ffmpeg_path`)
- Internet access to Hugging Face for the first-run model download and to
  iptv-org for the public channel catalog
- Several GB of free disk space for the Large-v3 model, Python environment,
  and cached media
- For the checked-in CUDA configuration: an NVIDIA GPU with a current driver
  compatible with CUDA 12; CPU mode is possible by changing the config to
  `device = "cpu"`

The launcher installs the Python dependencies and CUDA runtime wheels when
needed. If PowerShell script execution is restricted, run it in a temporary
process policy scope:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

```bash
./run-public-poc.sh
```

On Windows PowerShell, the all-in-one launcher performs the same virtual
environment and dependency setup:

```powershell
.\run-public-poc.ps1
```

Pass a custom config as the first argument, followed by any CLI arguments:

```powershell
.\run-public-poc.ps1 .\config.public.example.toml
```

It listens on `http://127.0.0.1:8081/` by default. Individual public streams
can be unavailable or use codecs that a browser cannot play; select another
channel when the catalog entry cannot be packaged. The catalog is downloaded
once to `.runtime/public-iptv/index.m3u`; later launches use that cached copy.

Open `http://127.0.0.1:8081/`. The service resolves channels from the public
catalog, packages the selected channel locally as CMAF with AAC-LC
audio, and sends finalized Large-v3 segments to the subtitle overlay via SSE.
The subtitle switch changes only display; local transcription remains active
for the selected channel.

The checked-in example targets the local RTX 4060 (`device = "cuda"`,
`compute_type = "float16"`). CUDA must be visible to the process; restricted
sandbox environments intentionally do not expose the GPU even when the normal
desktop shell does.
