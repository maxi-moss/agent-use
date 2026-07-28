"""Headless session broker entrypoint: `python -m broker.session`."""

import argparse
import asyncio

from broker import logging_setup
from broker.paths import BrokerPaths
from broker.session.broker import SessionBroker
from broker.config import SessionBrokerConfig


def main() -> None:
    """Parse ``--config-json`` and run the session broker until it exits."""
    parser = argparse.ArgumentParser(prog="broker.session")
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()
    cfg = SessionBrokerConfig.model_validate_json(args.config_json)
    logging_setup.configure(BrokerPaths(cfg.broker_home).session_log(cfg.name))
    asyncio.run(SessionBroker(cfg).run())


if __name__ == "__main__":
    main()
