"""File extraction: tree-sitter parse → symbols, as-written references, imports.

Pure: bytes in, ``FileExtraction`` out. One walker for every language; the
per-language differences live in ``languages`` as tables and hooks. Nested
functions, lambdas and classes inside functions are not symbols — they belong
to the enclosing symbol's body and their calls are attributed to it.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

from tree_sitter import Node, Parser

from broker.index.languages import LanguageSpec, dotted_name, name_of, node_text
from broker.index.schemas import (
    FileExtraction,
    Import,
    RefKind,
    Reference,
    Symbol,
    SymbolKind,
    qualified_name,
)

_FIRST_LINE_MAX = 120


@dataclass
class _Collected:
    symbols: dict[str, Symbol] = field(default_factory=dict[str, Symbol])
    references: set[tuple[str, RefKind, str]] = field(
        default_factory=set[tuple[str, RefKind, str]]
    )
    imports: list[Import] = field(default_factory=list[Import])


@dataclass(frozen=True)
class _File:
    spec: LanguageSpec
    path: str
    source: bytes

    def text(self, start: int, end: int) -> str:
        """Return the source between two byte offsets as text."""
        return self.source[start:end].decode("utf-8", errors="replace")

    def line(self, byte: int) -> int:
        """Return the 1-based line number of a byte offset."""
        return self.source.count(b"\n", 0, byte) + 1


def extract_file(spec: LanguageSpec, path: str, source: bytes) -> FileExtraction:
    """Extract every indexable symbol, reference and import from one file.

    Args:
        spec: Language tables for the file.
        path: Repository-relative path, used as the symbol name prefix.
        source: File contents.

    Returns:
        The extraction, with references deduplicated and sorted.
    """
    tree = Parser(spec.language).parse(source)
    file = _File(spec, path, source)
    acc = _Collected()
    for statement in _module_statements(spec, tree.root_node):
        _module_statement(file, statement, acc)
    return FileExtraction(
        path=path,
        symbols=list(acc.symbols.values()),
        references=[
            Reference(source=s, kind=k, target=t) for s, k, t in sorted(acc.references)
        ],
        imports=acc.imports,
    )


def _module_statements(spec: LanguageSpec, node: Node) -> Iterator[Node]:
    """Yield a module's top-level statements, flattening transparent wrappers."""
    for child in node.named_children:
        if child.type in spec.transparent_nodes:
            yield from _module_statements(spec, child)
        else:
            yield child


def _unwrap(spec: LanguageSpec, node: Node) -> Node | None:
    """Strip decorator/export wrappers and expression statements to the declaration."""
    current = node
    while True:
        field_name = spec.wrapper_nodes.get(current.type)
        if field_name is not None:
            inner = current.child_by_field_name(field_name)
            if inner is None:
                return None
            current = inner
            continue
        if current.type in spec.statement_wrappers:
            if not current.named_children:
                return None
            current = current.named_children[0]
            continue
        return current


def _module_statement(file: _File, node: Node, acc: _Collected) -> None:
    """Extract one top-level statement into symbols, imports or references."""
    spec = file.spec
    inner = _unwrap(spec, node)
    if inner is None:
        return
    start = node.start_byte
    if inner.type in spec.import_nodes:
        acc.imports.extend(spec.imports(inner, file.path))
    elif inner.type in spec.class_nodes:
        _class(file, node, inner, start, "", acc)
    elif inner.type in spec.function_nodes:
        name = name_of(inner, ("name",))
        if name is not None:
            _callable(file, node, inner, start, name, SymbolKind.FUNCTION, "", acc)
    elif inner.type in spec.type_nodes:
        _type(file, inner, start, acc)
    elif inner.type in spec.variable_nodes:
        _assigned_functions(file, node, inner, start, acc)


def _class(
    file: _File,
    outer: Node,
    cls: Node,
    outer_start: int,
    scope: str,
    acc: _Collected,
) -> None:
    """Extract a class, its members, bases and calls."""
    spec = file.spec
    name = name_of(cls, ("name",))
    body = cls.child_by_field_name("body")
    if name is None or body is None:
        return
    class_scope = f"{scope}.{name}" if scope else name
    class_qname = qualified_name(file.path, scope, name)
    fields: list[str] = []
    child_symbol_ids: set[int] = set()
    decorator_start: int | None = None
    for member in body.named_children:
        if member.type == spec.decorator_node:
            if decorator_start is None:
                decorator_start = member.start_byte
            continue
        start = decorator_start if decorator_start is not None else member.start_byte
        decorator_start = None
        _member(file, member, start, class_scope, class_qname, fields, child_symbol_ids, acc)
    _merge(
        acc,
        Symbol(
            path=file.path,
            scope=scope,
            name=name,
            kind=SymbolKind.CLASS,
            start_line=file.line(outer_start),
            end_line=file.line(cls.end_byte),
            signature=file.text(outer_start, body.start_byte).rstrip(),
            docstring=spec.docstring(outer, cls),
            body=file.text(body.start_byte, body.end_byte),
            fields=fields,
        ),
    )
    for base in spec.bases(cls):
        acc.references.add((class_qname, RefKind.BASE, base))
    for target in _calls(spec, outer, frozenset(child_symbol_ids)):
        acc.references.add((class_qname, RefKind.CALL, target))


def _member(
    file: _File,
    node: Node,
    start: int,
    class_scope: str,
    class_qname: str,
    fields: list[str],
    child_symbol_ids: set[int],
    acc: _Collected,
) -> None:
    """Extract one class member: method, nested class, field or function-valued field."""
    spec = file.spec
    inner = _unwrap(spec, node)
    if inner is None:
        return
    if inner.type in spec.method_nodes:
        name = name_of(inner, ("name",))
        if name is not None:
            child_symbol_ids.add(node.id)
            _callable(file, node, inner, start, name, SymbolKind.METHOD, class_scope, acc)
    elif inner.type in spec.class_nodes:
        child_symbol_ids.add(node.id)
        _class(file, node, inner, start, class_scope, acc)
    elif inner.type in spec.field_nodes:
        name = name_of(inner, spec.field_name_fields)
        if name is None:
            return
        _, value_field, type_field = spec.declarator_fields
        value = inner.child_by_field_name(value_field)
        annotation = inner.child_by_field_name(type_field)
        if value is not None and value.type in spec.function_value_nodes:
            child_symbol_ids.add(node.id)
            _callable(file, node, value, start, name, SymbolKind.METHOD, class_scope, acc)
            return
        if annotation is None:
            if not spec.fields_require_annotation:
                fields.append(name)
            return
        fields.append(f"{name}: {_annotation_text(annotation)}")
        for target in _type_names(spec, annotation):
            acc.references.add((class_qname, RefKind.TYPE, target))


def _callable(
    file: _File,
    outer: Node,
    fn: Node,
    outer_start: int,
    name: str,
    kind: SymbolKind,
    scope: str,
    acc: _Collected,
) -> None:
    """Record a function or method: header, docstring, body, calls, annotation types."""
    spec = file.spec
    body = fn.child_by_field_name("body")
    header_end = body.start_byte if body is not None else fn.end_byte
    signature = file.text(outer_start, header_end).rstrip().rstrip(";").rstrip()
    symbol = Symbol(
        path=file.path,
        scope=scope,
        name=name,
        kind=kind,
        start_line=file.line(outer_start),
        end_line=file.line(fn.end_byte),
        signature=signature,
        docstring=spec.docstring(outer, fn),
        body=file.text(body.start_byte, body.end_byte) if body is not None else "",
    )
    _merge(acc, symbol)
    qname = symbol.qualified_name
    for target in _calls(spec, outer, frozenset()):
        acc.references.add((qname, RefKind.CALL, target))
    for target in _annotation_names(spec, fn, skip=body):
        acc.references.add((qname, RefKind.TYPE, target))


def _type(file: _File, inner: Node, outer_start: int, acc: _Collected) -> None:
    """Extract a type-alias, interface or enum symbol."""
    name = name_of(inner, ("name",))
    if name is None:
        return
    body = inner.child_by_field_name("body")
    if body is not None:
        signature = file.text(outer_start, body.start_byte).rstrip()
    else:
        signature = _first_line(file.text(outer_start, inner.end_byte))
    _merge(
        acc,
        Symbol(
            path=file.path,
            scope="",
            name=name,
            kind=SymbolKind.TYPE,
            start_line=file.line(outer_start),
            end_line=file.line(inner.end_byte),
            signature=signature,
        ),
    )


def _assigned_functions(
    file: _File, outer: Node, inner: Node, outer_start: int, acc: _Collected
) -> None:
    """Record functions assigned in a declaration (``const f = () => ...``)."""
    spec = file.spec
    name_field, value_field, _ = spec.declarator_fields
    if inner.type == spec.declarator_node:
        declarators = [inner]
    else:
        declarators = [c for c in inner.named_children if c.type == spec.declarator_node]
    for decl in declarators:
        name_node = decl.child_by_field_name(name_field)
        if name_node is None or name_node.type != "identifier":
            continue
        value = decl.child_by_field_name(value_field)
        if value is not None and value.type in spec.function_value_nodes:
            _callable(
                file, outer, value, outer_start, node_text(name_node),
                SymbolKind.FUNCTION, "", acc,
            )


def _merge(acc: _Collected, symbol: Symbol) -> None:
    """Collapse overloads and redefinitions into one symbol per qualified name."""
    existing = acc.symbols.get(symbol.qualified_name)
    if existing is None:
        acc.symbols[symbol.qualified_name] = symbol
        return
    acc.symbols[symbol.qualified_name] = existing.model_copy(
        update={
            "signature": existing.signature + "\n" + symbol.signature,
            "docstring": symbol.docstring or existing.docstring,
            "body": symbol.body or existing.body,
            "end_line": symbol.end_line,
            "fields": existing.fields + symbol.fields,
        }
    )


def _calls(spec: LanguageSpec, root: Node, prune: frozenset[int]) -> list[str]:
    """Collect dotted callee names under ``root``, skipping pruned subtrees."""
    out: list[str] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.id in prune:
            continue
        callee_field = spec.call_nodes.get(node.type)
        if callee_field is not None:
            callee = node.child_by_field_name(callee_field)
            if callee is not None:
                target = dotted_name(callee, spec.member_access_node)
                if target is not None:
                    out.append(target)
        stack.extend(node.named_children)
    return out


def _annotation_names(spec: LanguageSpec, root: Node, skip: Node | None) -> set[str]:
    """Collect type names from every annotation under ``root`` except ``skip``."""
    names: set[str] = set()
    skip_id = skip.id if skip is not None else None
    stack = [root]
    while stack:
        node = stack.pop()
        if node.id == skip_id:
            continue
        if node.type in spec.annotation_nodes:
            names.update(_type_names(spec, node))
            continue
        stack.extend(node.named_children)
    return names


def _type_names(spec: LanguageSpec, root: Node) -> set[str]:
    """Collect the type names referenced anywhere under ``root``."""
    names: set[str] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in spec.type_name_nodes:
            text = dotted_name(node, spec.member_access_node) or node_text(node)
            if text:
                names.add(text)
            continue
        stack.extend(node.named_children)
    return names


def _annotation_text(node: Node) -> str:
    """Return an annotation node's text without its leading colon."""
    return node_text(node).lstrip(":").strip()


def _first_line(text: str) -> str:
    """Return the first line, truncated, for a bodyless type signature."""
    first, _, rest = text.partition("\n")
    first = first.rstrip()
    if len(first) > _FIRST_LINE_MAX or rest:
        return first[:_FIRST_LINE_MAX].rstrip() + " ..."
    return first
