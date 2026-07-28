"""Headless session broker entrypoint: `python -m broker.session`."""

import argparse
import asyncio
import logging

from broker.session.broker import SessionBroker
from broker.session.config import SessionBrokerConfig


def main() -> None:
    """Parse ``--config-json`` and run the session broker until it exits."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(prog="broker.session")
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()
    cfg = SessionBrokerConfig.model_validate_json(args.config_json)
    asyncio.run(SessionBroker(cfg).run())


if __name__ == "__main__":
    main()
