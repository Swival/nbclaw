"""Entry point: ``python -m nbclaw`` / ``nbclaw``."""

from __future__ import annotations

import asyncio
import logging
import os

from .config import build_config
from .daemon import Daemon


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("NBCLAW_LOG", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = build_config()
    daemon = Daemon(config)
    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
