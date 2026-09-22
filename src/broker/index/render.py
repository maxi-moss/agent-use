"""Render a GroundingContext as the ``# Relevant code`` block.

Signatures and edge lines only — no docstrings, no bodies. Deterministic:
the same context renders byte-identically.
"""

from broker.index.schemas import ContextSymbol, EdgeKind, GroundingContext, SymbolKind

BUDGET_CHARS = 16_000  # ~4k tokens at ~4 chars/token; deliberately no tokenizer
_HEADER = "# Relevant code"


def fit_to_budget(context: GroundingContext) -> GroundingContext:
    """Drop the lowest-ranked expansion symbols until the block fits the budget.

    Seeds are never dropped; a seeds-only block may exceed the budget.
    """
    kept = list(context.symbols)
    while (
        len(render_relevant_code(context.model_copy(update={"symbols": kept})))
        > BUDGET_CHARS
    ):
        expansion = [i for i, s in enumerate(kept) if s.score is None]
        if not expansion:
            break
        del kept[expansion[-1]]
    return context.model_copy(update={"symbols": kept})


def render_relevant_code(context: GroundingContext) -> str:
    """Render the block: grouped by file, seeds first, methods nested under their class."""
    present = {s.qualified_name for s in context.symbols}
    by_path: dict[str, list[ContextSymbol]] = {}
    for symbol in context.symbols:
        by_path.setdefault(symbol.path, []).append(symbol)
    lines = [_HEADER]
    for path, symbols in by_path.items():
        lines += ["", f"## {path}"]
        imports = context.imports.get(path)
        if imports:
            lines.append("imports: " + ", ".join(imports))
        nested = {
            s.qualified_name
            for s in symbols
            if s.kind is SymbolKind.METHOD and _owner(s.qualified_name) in present
        }
        for symbol in symbols:
            if symbol.qualified_name in nested:
                continue
            lines += _symbol_lines(context, symbol, indent="")
            if symbol.kind is SymbolKind.CLASS:
                for method in symbols:
                    if (
                        method.qualified_name in nested
                        and _owner(method.qualified_name) == symbol.qualified_name
                    ):
                        lines += _symbol_lines(context, method, indent="  ")
    return "\n".join(lines)


def _owner(qualified_name: str) -> str | None:
    """Return the qualified name of a symbol's owning class, or ``None``."""
    path, _, inner = qualified_name.partition("::")
    if "." not in inner:
        return None
    return f"{path}::{inner.rsplit('.', 1)[0]}"


def _symbol_lines(
    context: GroundingContext, symbol: ContextSymbol, indent: str
) -> list[str]:
    """Render one symbol's heading, signature and edge lines."""
    _, _, inner = symbol.qualified_name.partition("::")
    tag = symbol.kind.value + (", seed" if symbol.score is not None else "")
    lines = [
        f"{indent}### {inner} ({tag}, lines {symbol.start_line}-{symbol.end_line})"
    ]
    lines += [f"{indent}    {line}" for line in symbol.signature.splitlines()]
    if symbol.kind is SymbolKind.CLASS:
        lines += [f"{indent}    {field}" for field in symbol.fields]
        if symbol.methods:
            lines.append(f"{indent}  methods: {', '.join(symbol.methods)}")
    q = symbol.qualified_name
    edges = context.edges
    groups = [
        (
            "calls",
            sorted(
                {e.target for e in edges if e.kind is EdgeKind.CALLS and e.source == q}
            ),
        ),
        (
            "called by",
            sorted(
                {e.source for e in edges if e.kind is EdgeKind.CALLS and e.target == q}
            ),
        ),
        (
            "owner",
            sorted(
                {
                    e.source
                    for e in edges
                    if e.kind is EdgeKind.DEFINES and e.target == q
                }
            ),
        ),
        (
            "bases",
            sorted(
                {
                    e.target
                    for e in edges
                    if e.kind is EdgeKind.INHERITS and e.source == q
                }
            ),
        ),
        (
            "types",
            sorted(
                {
                    e.target
                    for e in edges
                    if e.kind is EdgeKind.REFERENCES_TYPE and e.source == q
                }
            ),
        ),
    ]
    for label, names in groups:
        if names:
            lines.append(f"{indent}  {label}: {', '.join(names)}")
    return lines
