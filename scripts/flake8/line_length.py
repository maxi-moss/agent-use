"""flake8 plugin: pycodestyle's E501 with docstring lines exempt."""

import ast
import tokenize
from collections.abc import Iterator

import pycodestyle

_DOCUMENTED = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _docstring_rows(tree: ast.AST) -> set[int]:
    """Return the line numbers docstrings occupy."""
    rows: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _DOCUMENTED) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            rows.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return rows


def _multiline_string_rows(tokens: list[tokenize.TokenInfo]) -> set[int]:
    """Return the line numbers spanned by multi-line string tokens."""
    rows: set[int] = set()
    for tok in tokens:
        if tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
            rows.update(range(tok.start[0], tok.end[0] + 1))
    return rows


class LineLength:
    """Report E501 as BLL501 on every line outside a docstring."""

    name = "broker-line-length"
    version = "1"

    def __init__(
        self,
        tree: ast.AST,
        lines: list[str],
        file_tokens: list[tokenize.TokenInfo],
        max_line_length: int,
    ) -> None:
        self._tree = tree
        self._lines = lines
        self._tokens = file_tokens
        self._max_line_length = max_line_length

    def run(self) -> Iterator[tuple[int, int, str, type["LineLength"]]]:
        """Yield one BLL501 per overlong non-docstring line."""
        exempt = _docstring_rows(self._tree)
        multiline = _multiline_string_rows(self._tokens)
        for row, line in enumerate(self._lines, start=1):
            if row in exempt:
                continue
            hit = pycodestyle.maximum_line_length(
                line, self._max_line_length, row in multiline, row, False
            )
            if hit is not None:
                offset, text = hit
                yield row, offset, "BLL" + text.removeprefix("E"), type(self)
