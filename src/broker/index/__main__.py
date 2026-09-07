"""Index CLI: ``python -m broker.index <repo>``.

One summary line on stdout on success; one error line on stderr and a
non-zero exit on failure. Everything else goes to the index log file.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import NoReturn

from broker import config as broker_config
from broker import logging_setup
from broker.index.embedding import EmbeddingError, OpenAIEmbedder
from broker.index.indexer import IndexingError, index_repo
from broker.paths import BrokerPaths


def _fail(reason: str) -> NoReturn:
    print(f"broker index: {reason}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="broker.index")
    parser.add_argument("repo", help="root of the git repository to index")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    try:
        cfg = broker_config.load()
    except broker_config.ConfigError as exc:
        _fail(str(exc))
    paths = BrokerPaths(cfg.broker_home)
    logging_setup.configure(paths.index_log)
    try:
        embedder = OpenAIEmbedder.from_env(cfg.embedding)
        summary = asyncio.run(index_repo(repo, paths.index_db(repo), embedder))
    except (IndexingError, EmbeddingError) as exc:
        _fail(str(exc))
    print(
        f"indexed {repo}: {summary.files_changed} files changed, "
        f"{summary.symbols} symbols, {summary.edges} edges, "
        f"{summary.embedded} embedded"
    )


if __name__ == "__main__":
    main()
