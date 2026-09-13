"""Download the configured default Whisper model when it is not local."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

MODEL_REPOSITORY = "Systran/faster-whisper-large-v3"


def model_path(config_path: Path) -> Path:
    with config_path.open("rb") as file:
        document = tomllib.load(file)
    value = document["model"]["model_path"]
    path = Path(value).expanduser()
    return path if path.is_absolute() else config_path.parent / path


def ensure_model(config_path: Path) -> Path:
    path = model_path(config_path)
    if (path / "model.bin").is_file():
        print(f"Using local Whisper model: {path}")
        return path

    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_REPOSITORY} to {path} ...")
    path.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_REPOSITORY, local_dir=str(path))
    if not (path / "model.bin").is_file():
        raise RuntimeError(f"model download completed without model.bin: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensure the configured Faster-Whisper model is local")
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    ensure_model(Path(arguments.config).expanduser().resolve())


if __name__ == "__main__":
    main()
