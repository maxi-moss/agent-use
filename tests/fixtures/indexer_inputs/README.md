# Indexer inputs — test data, not project code

The directories here are small sample repositories that `broker.index` parses in `tests/unit/test_index_symbol_extraction.py`, `tests/unit/test_index_indexer.py` and `tests/unit/test_index_embedding.py`. The indexer supports Python, TypeScript, TSX and JavaScript, so `typescript_repo/` holds TypeScript and JavaScript files purely as parser input. This project's own code is Python only.

Nothing under this directory is imported, type-checked, linted or run. It is excluded from pyright and flake8 (`pyproject.toml`, `.flake8`) and marked `linguist-vendored` in `.gitattributes` so it does not count toward the repository's language statistics. Tests copy a directory into a temp dir and `git init` it before indexing, because the indexer refuses anything but a git top-level.

The files are deliberately shaped to exercise specific extraction and resolution cases (ambiguous method names, relative and module imports, overload stacks, arrow-function methods, a JavaScript file importing a TypeScript one), so the tests assert on their exact contents. Change them only together with those assertions.
