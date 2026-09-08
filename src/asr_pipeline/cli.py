from __future__ import annotations

import argparse
import asyncio
import sys

from .config import ConfigurationError, load_config
from .pipeline import CapacityError, Pipeline
from .sink import NdjsonStdoutSink


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real-time IPTV ASR pipeline")
    parser.add_argument("--config", required=True, help="path to the external TOML configuration")
    arguments = parser.parse_args()
    try:
        config = load_config(arguments.config)
        asyncio.run(Pipeline(config, NdjsonStdoutSink()).run())
    except (ConfigurationError, CapacityError) as error:
        print(f"configuration/startup error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
