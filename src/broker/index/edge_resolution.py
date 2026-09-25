"""Global edge resolution over the whole indexed repository.

Runs after every incremental parse, over every file, so edges are always
consistent with the current symbol set. As-written references that resolve to
nothing in the repository are dropped — only resolved edges exist.

Resolution order for a reference ``a.b.c`` written inside symbol S in file P:
same-file ``P::a.b.c``; then ``self``/``cls``/``this`` on S's class and its
indexed bases; then P's imports. Unresolved references are dropped.
"""

from collections import defaultdict, deque
from collections.abc import Iterable
from typing import assert_never

from broker.index.languages import (
    ECMA_MODULE_PROBES,
    SPECS_BY_NAME,
    LanguageSpec,
    python_module_names,
)
from broker.index.schemas import (
    Edge,
    EdgeKind,
    Import,
    RefKind,
    Reference,
    SymbolKey,
    SymbolKind,
    join_qualified_name,
    split_qualified_name,
)

_CALL_TARGETS = frozenset({SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.CLASS})
_TYPE_TARGETS = frozenset({SymbolKind.CLASS, SymbolKind.TYPE})


def resolve_edges(
    symbols: Iterable[SymbolKey],
    references: Iterable[Reference],
    imports: Iterable[Import],
    languages: dict[str, str],
) -> list[Edge]:
    """Resolve every stored reference and import into edges.

    Args:
        symbols: Every symbol in the index.
        references: Every as-written reference in the index.
        imports: Every import in the index.
        languages: Indexed file path → language name.

    Returns:
        Deduplicated edges, sorted by (source, target, kind).
    """
    return _Resolver(symbols, imports, languages).edges(references)


class _Resolver:
    def __init__(
        self,
        symbols: Iterable[SymbolKey],
        imports: Iterable[Import],
        languages: dict[str, str],
    ) -> None:
        self._specs: dict[str, LanguageSpec] = {
            path: SPECS_BY_NAME[name] for path, name in languages.items()
        }
        self._by_qname: dict[str, SymbolKey] = {}
        for symbol in symbols:
            self._by_qname[symbol.qualified_name] = symbol
        self._imports = {(i.path, i.local_name): i for i in imports}
        self._modules: dict[str, list[str]] = defaultdict(list)
        for path, spec in self._specs.items():
            if spec.import_resolution == "dotted_module":
                for dotted in python_module_names(path):
                    self._modules[dotted].append(path)
        self._bases: dict[str, list[str]] = defaultdict(list)

    def edges(self, references: Iterable[Reference]) -> list[Edge]:
        """Resolve all references, imports and definitions into deduplicated edges."""
        found: set[tuple[str, str, EdgeKind]] = set()
        ordered = sorted(references, key=lambda r: (r.source, r.kind, r.target))
        # Bases first: self/this resolution walks the class hierarchy.
        for ref in ordered:
            if ref.kind is RefKind.BASE:
                target = self._resolve(
                    ref.source, ref.target, _TYPE_TARGETS, hierarchy=False
                )
                if target is not None:
                    found.add((ref.source, target, EdgeKind.INHERITS))
                    self._bases[ref.source].append(target)
        for ref in ordered:
            if ref.kind is RefKind.CALL:
                target = self._resolve(
                    ref.source, ref.target, _CALL_TARGETS, hierarchy=True
                )
                if target is not None:
                    found.add((ref.source, target, EdgeKind.CALLS))
            elif ref.kind is RefKind.TYPE:
                target = self._resolve(
                    ref.source, ref.target, _TYPE_TARGETS, hierarchy=False
                )
                if target is not None:
                    found.add((ref.source, target, EdgeKind.REFERENCES_TYPE))
        for imp in self._imports.values():
            target = self._via_import(imp, [])
            if target is not None:
                found.add((imp.path, target, EdgeKind.IMPORTS))
        for symbol in self._by_qname.values():
            if symbol.scope:
                found.add(
                    (
                        join_qualified_name(symbol.path, symbol.scope),
                        symbol.qualified_name,
                        EdgeKind.DEFINES,
                    )
                )
        return [Edge(source=s, target=t, kind=k) for s, t, k in sorted(found)]

    def _is(self, qname: str, kinds: frozenset[SymbolKind]) -> bool:
        """Report whether ``qname`` names an indexed symbol of one of ``kinds``."""
        symbol = self._by_qname.get(qname)
        return symbol is not None and symbol.kind in kinds

    def _resolve(
        self, source: str, target: str, kinds: frozenset[SymbolKind], *, hierarchy: bool
    ) -> str | None:
        """Resolve one as-written reference to a target symbol's qualified name, or ``None``."""
        path, inner = split_qualified_name(source)
        parts = target.split(".")
        candidate = join_qualified_name(path, target)
        if self._is(candidate, kinds):
            return candidate
        self_names = self._specs[path].self_names
        if hierarchy and len(parts) > 1 and parts[0] in self_names:
            owner_inner = self._owner_inner(source, inner)
            if owner_inner:
                rest = ".".join(parts[1:])
                for cls in self._hierarchy(join_qualified_name(path, owner_inner)):
                    candidate = f"{cls}.{rest}"
                    if self._is(candidate, kinds):
                        return candidate
        imp = self._imports.get((path, parts[0]))
        if imp is not None:
            resolved = self._via_import(imp, parts[1:])
            if resolved is not None and self._is(resolved, kinds):
                return resolved
        return None

    def _owner_inner(self, source: str, inner: str) -> str:
        """The class that ``self``/``this`` means inside ``source``."""
        symbol = self._by_qname.get(source)
        if symbol is not None and symbol.kind is SymbolKind.CLASS:
            return inner
        return inner.rpartition(".")[0]

    def _hierarchy(self, cls: str) -> list[str]:
        """Return a class and its indexed bases, nearest first."""
        seen = [cls]
        queue = deque([cls])
        while queue:
            current = queue.popleft()
            for base in self._bases.get(current, []):
                if base not in seen:
                    seen.append(base)
                    queue.append(base)
        return seen

    def _via_import(self, imp: Import, rest: list[str]) -> str | None:
        """Resolve an import binding plus a trailing member-access chain to a symbol or module."""
        resolution = self._specs[imp.path].import_resolution
        if resolution == "dotted_module":
            return self._via_dotted_module(imp, rest)
        elif resolution == "file_path":
            return self._via_file_path(imp, rest)
        else:
            assert_never(resolution)

    def _via_dotted_module(self, imp: Import, rest: list[str]) -> str | None:
        """Resolve an import addressed by dotted module name to a symbol or module."""
        dotted = (
            imp.module.split(".")
            + ([imp.imported_name] if imp.imported_name else [])
            + rest
        )
        for cut in range(len(dotted), 0, -1):
            paths = self._modules.get(".".join(dotted[:cut]), [])
            if len(paths) != 1:
                continue
            remaining = dotted[cut:]
            if not remaining:
                return paths[0]
            candidate = join_qualified_name(paths[0], ".".join(remaining))
            return candidate if candidate in self._by_qname else None
        return None

    def _via_file_path(self, imp: Import, rest: list[str]) -> str | None:
        """Resolve an import addressed by relative file path to a symbol or module."""
        module_path = self._module_path(imp.module)
        if module_path is None:
            return None
        chain = (
            [imp.imported_name] if imp.imported_name not in ("", "default") else []
        ) + rest
        if not chain:
            return module_path
        candidate = join_qualified_name(module_path, ".".join(chain))
        return candidate if candidate in self._by_qname else None

    def _module_path(self, module: str) -> str | None:
        """Resolve a relative module specifier to an indexed file path."""
        if module in self._specs:
            return module
        for probe in ECMA_MODULE_PROBES:
            candidate = module + probe
            if candidate in self._specs:
                return candidate
        return None
