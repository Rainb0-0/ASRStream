from __future__ import annotations

import argparse
import sys

from .config import ConfigurationError
from .poc import PocService, PublicPlaylistClient, load_public_poc_config
from .poc_cli import serve


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the public IPTV ASR proof-of-concept player")
    parser.add_argument("--config", required=True, help="path to the public POC TOML configuration")
    arguments = parser.parse_args()
    try:
        config = load_public_poc_config(arguments.config)
        service = PocService(config, PublicPlaylistClient(config))
    except (ConfigurationError, RuntimeError) as error:
        print(f"Public POC startup error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    serve(config, service, "Public IPTV POC player")


if __name__ == "__main__":
    main()
