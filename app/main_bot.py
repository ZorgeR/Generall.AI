"""GALL.AI Telegram bot entrypoint.

Run from the ``app/`` directory: ``python main_bot.py``. Startup order matters:

1. ``.env`` is loaded and configuration validated.
2. The secure-container system is initialised; this monkey-patches the tool
   classes so sandboxed tools run in Docker.
3. ``agents.main`` is imported only afterwards, so every agent sees the
   patched classes.
4. The aiogram application starts polling.
"""
from __future__ import annotations

import asyncio
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    from bot.config import config

    problems = config.validate()
    if problems:
        for p in problems:
            logger.error(p)
        sys.exit(1)

    from secure_container.main import cleanup_containers, initialize_secure_containers

    if not initialize_secure_containers():
        logger.error("Secure container system initialization failed!")
        sys.exit(1)
    logger.info("Secure container system initialized successfully!")

    import agents.main  # noqa: F401  (must come after container initialisation)

    from bot.app import run

    try:
        asyncio.run(run())
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cleanup_containers()


if __name__ == "__main__":
    main()
