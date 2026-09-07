"""Per-language extraction tables.

A ``LanguageSpec`` is data — which tree-sitter node types play which structural
role and which field holds a name — plus the three hooks whose shape genuinely
differs between grammars (imports, docstrings, base classes). ``extract`` holds
the single walker; adding a language means adding a spec.
"""

import posixpath
import re
from collections.abc import Callable
from dataclasses import dataclass

import tree_sitter_javascript
import tree_sitter_python
import tree_sitter_typescript
from tree_sitter import Language, Node

from broker.index.schemas import Import

_DOTTED_RE = re.compile(r"^[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*$")
_ECMA_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
_NAME_NODE_TYPES = frozenset(
    {"identifier", "property_identifier", "type_identifier"}
)


def node_text(node: Node) -> str:
    """Return a node's source text; tree-sitter yields bytes."""
    return (node.text or b"").decode("utf-8", errors="replace")


def dotted_name(node: Node, member_access_node: str) -> str | None:
    """Return ``a.b.c`` for an identifier or plain member-access chain, else ``None``."""
    if node.type not in ("identifier", member_access_node):
        return None
    text = node_text(node)
    return text if _DOTTED_RE.match(text) else None


def name_of(node: Node, fields: tuple[str, ...]) -> str | None:
    """Return the identifier text held by the first present field in ``fields``."""
    for field_name in fields:
        child = node.child_by_field_name(field_name)
        if child is not None and child.type in _NAME_NODE_TYPES:
            return node_text(child)
    return None


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    extensions: frozenset[str]
    language: Language
    wrapper_nodes: dict[str, str]
    statement_wrappers: frozenset[str]
    transparent_nodes: frozenset[str]
    function_nodes: frozenset[str]
    class_nodes: frozenset[str]
    method_nodes: frozenset[str]
    type_nodes: frozenset[str]
    field_nodes: frozenset[str]
    field_name_fields: tuple[str, ...]
    fields_require_annotation: bool
    variable_nodes: frozenset[str]
    declarator_node: str
    declarator_fields: tuple[str, str, str]
    function_value_nodes: frozenset[str]
    import_nodes: frozenset[str]
    call_nodes: dict[str, str]
    member_access_node: str
    annotation_nodes: frozenset[str]
    type_name_nodes: frozenset[str]
    decorator_node: str
    imports: Callable[[Node, str], list[Import]]
    docstring: Callable[[Node, Node], str]
    bases: Callable[[Node], list[str]]


# ── Python hooks ──────────────────────────────────────────────────────────


def _python_package(path: str) -> list[str]:
    """Dotted package holding ``path``; the directory, for modules and __init__ alike."""
    return path[: -len(".py")].split("/")[:-1]


def _python_imports(node: Node, path: str) -> list[Import]:
    """Extract Python import bindings from an import statement."""
    out: list[Import] = []
    if node.type == "import_statement":
        for child in node.named_children:
            if child.type == "dotted_name":
                first = node_text(child).split(".")[0]
                out.append(
                    Import(path=path, local_name=first, module=first, imported_name="")
                )
            elif child.type == "aliased_import":
                name = child.child_by_field_name("name")
                alias = child.child_by_field_name("alias")
                if name is not None and alias is not None:
                    out.append(
                        Import(
                            path=path,
                            local_name=node_text(alias),
                            module=node_text(name),
                            imported_name="",
                        )
                    )
        return out
    module_node = node.child_by_field_name("module_name")
    if module_node is None:
        return out
    if module_node.type == "relative_import":
        dots = 0
        tail = ""
        for child in module_node.children:
            if child.type == "import_prefix":
                dots = len(node_text(child))
            elif child.type == "dotted_name":
                tail = node_text(child)
        package = _python_package(path)
        base = package[: max(0, len(package) - (dots - 1))]
        module = ".".join(base + ([tail] if tail else []))
    else:
        module = node_text(module_node)
    for child in node.children_by_field_name("name"):
        if child.type == "dotted_name":
            imported_name = node_text(child)
            out.append(
                Import(
                    path=path, local_name=imported_name, module=module,
                    imported_name=imported_name,
                )
            )
        elif child.type == "aliased_import":
            name = child.child_by_field_name("name")
            alias = child.child_by_field_name("alias")
            if name is not None and alias is not None:
                out.append(
                    Import(
                        path=path,
                        local_name=node_text(alias),
                        module=module,
                        imported_name=node_text(name),
                    )
                )
    return out


def _python_docstring(outer: Node, inner: Node) -> str:
    """Return the leading string literal of a Python definition's body."""
    del outer
    body = inner.child_by_field_name("body")
    if body is None or not body.named_children:
        return ""
    first = body.named_children[0]
    if first.type != "expression_statement" or not first.named_children:
        return ""
    string = first.named_children[0]
    if string.type != "string":
        return ""
    content = [c for c in string.named_children if c.type == "string_content"]
    return node_text(content[0]).strip() if content else ""


def _python_bases(class_node: Node) -> list[str]:
    """Return the base-class names of a Python class."""
    supers = class_node.child_by_field_name("superclasses")
    if supers is None:
        return []
    out: list[str] = []
    for child in supers.named_children:
        candidate = child
        if child.type == "subscript":
            value = child.child_by_field_name("value")
            if value is not None:
                candidate = value
        name = dotted_name(candidate, "attribute")
        if name is not None:
            out.append(name)
    return out


# ── TypeScript / TSX / JavaScript hooks ───────────────────────────────────


def _ecma_module(path: str, specifier: str) -> str:
    """Normalise a relative specifier to a repo path without extension; keep bare ones."""
    if not specifier.startswith("."):
        return specifier
    joined = posixpath.normpath(posixpath.join(posixpath.dirname(path), specifier))
    for ext in _ECMA_EXTENSIONS:
        if joined.endswith(ext):
            return joined[: -len(ext)]
    return joined


def _ecma_imports(node: Node, path: str) -> list[Import]:
    """Extract import bindings from an ECMAScript import statement."""
    source = node.child_by_field_name("source")
    if source is None:
        return []
    module = _ecma_module(path, node_text(source).strip("'\"`"))
    out: list[Import] = []
    for clause in (c for c in node.named_children if c.type == "import_clause"):
        for child in clause.named_children:
            if child.type == "identifier":
                out.append(
                    Import(
                        path=path,
                        local_name=node_text(child),
                        module=module,
                        imported_name="default",
                    )
                )
            elif child.type == "namespace_import":
                ids = [c for c in child.named_children if c.type == "identifier"]
                if ids:
                    out.append(
                        Import(
                            path=path,
                            local_name=node_text(ids[0]),
                            module=module,
                            imported_name="",
                        )
                    )
            elif child.type == "named_imports":
                for spec in child.named_children:
                    if spec.type != "import_specifier":
                        continue
                    name = spec.child_by_field_name("name")
                    alias = spec.child_by_field_name("alias")
                    if name is None:
                        continue
                    local = alias if alias is not None else name
                    out.append(
                        Import(
                            path=path,
                            local_name=node_text(local),
                            module=module,
                            imported_name=node_text(name),
                        )
                    )
    return out


def _ecma_docstring(outer: Node, inner: Node) -> str:
    """Return the ``/** */`` block immediately preceding a declaration."""
    del inner
    prev = outer.prev_named_sibling
    if prev is None or prev.type != "comment":
        return ""
    text = node_text(prev)
    return text if text.startswith("/**") else ""


def _ecma_bases(class_node: Node) -> list[str]:
    """Return the extended and implemented type names of a class."""
    out: list[str] = []
    for heritage in (
        c for c in class_node.named_children if c.type == "class_heritage"
    ):
        for clause in heritage.named_children:
            if clause.type == "extends_clause":
                value = clause.child_by_field_name("value")
                if value is not None:
                    name = dotted_name(value, "member_expression")
                    if name is not None:
                        out.append(name)
            elif clause.type == "implements_clause":
                out.extend(
                    node_text(t)
                    for t in clause.named_children
                    if t.type in ("type_identifier", "nested_type_identifier")
                )
            else:
                # JavaScript: class_heritage holds the expression directly.
                name = dotted_name(clause, "member_expression")
                if name is not None:
                    out.append(name)
    return out


# ── Specs ─────────────────────────────────────────────────────────────────

PYTHON = LanguageSpec(
    name="python",
    extensions=frozenset({".py"}),
    language=Language(tree_sitter_python.language()),
    wrapper_nodes={"decorated_definition": "definition"},
    statement_wrappers=frozenset({"expression_statement"}),
    transparent_nodes=frozenset(
        {
            "if_statement",
            "elif_clause",
            "else_clause",
            "try_statement",
            "except_clause",
            "finally_clause",
            "block",
        }
    ),
    function_nodes=frozenset({"function_definition"}),
    class_nodes=frozenset({"class_definition"}),
    method_nodes=frozenset({"function_definition"}),
    type_nodes=frozenset(),
    field_nodes=frozenset({"assignment"}),
    field_name_fields=("left",),
    fields_require_annotation=True,
    variable_nodes=frozenset(),
    declarator_node="assignment",
    declarator_fields=("left", "right", "type"),
    function_value_nodes=frozenset(),
    import_nodes=frozenset({"import_statement", "import_from_statement"}),
    call_nodes={"call": "function"},
    member_access_node="attribute",
    annotation_nodes=frozenset({"type"}),
    type_name_nodes=frozenset({"identifier", "attribute"}),
    decorator_node="decorator",
    imports=_python_imports,
    docstring=_python_docstring,
    bases=_python_bases,
)


def _ecma(name: str, extensions: frozenset[str], language: Language) -> LanguageSpec:
    """Build a ``LanguageSpec`` for an ECMAScript-family grammar."""
    return LanguageSpec(
        name=name,
        extensions=extensions,
        language=language,
        wrapper_nodes={"export_statement": "declaration"},
        statement_wrappers=frozenset(),
        transparent_nodes=frozenset(),
        function_nodes=frozenset(
            {"function_declaration", "generator_function_declaration"}
        ),
        class_nodes=frozenset({"class_declaration", "abstract_class_declaration"}),
        method_nodes=frozenset(
            {"method_definition", "abstract_method_signature", "method_signature"}
        ),
        type_nodes=frozenset(
            {"interface_declaration", "type_alias_declaration", "enum_declaration"}
        ),
        field_nodes=frozenset({"public_field_definition", "field_definition"}),
        field_name_fields=("name", "property"),
        fields_require_annotation=False,
        variable_nodes=frozenset({"lexical_declaration", "variable_declaration"}),
        declarator_node="variable_declarator",
        declarator_fields=("name", "value", "type"),
        function_value_nodes=frozenset(
            {"arrow_function", "function_expression", "function", "generator_function"}
        ),
        import_nodes=frozenset({"import_statement"}),
        call_nodes={"call_expression": "function", "new_expression": "constructor"},
        member_access_node="member_expression",
        annotation_nodes=frozenset({"type_annotation"}),
        type_name_nodes=frozenset({"type_identifier", "nested_type_identifier"}),
        decorator_node="decorator",
        imports=_ecma_imports,
        docstring=_ecma_docstring,
        bases=_ecma_bases,
    )


TYPESCRIPT = _ecma(
    "typescript",
    frozenset({".ts"}),
    Language(tree_sitter_typescript.language_typescript()),
)
TSX = _ecma(
    "tsx", frozenset({".tsx"}), Language(tree_sitter_typescript.language_tsx())
)
JAVASCRIPT = _ecma(
    "javascript",
    frozenset({".js", ".jsx", ".mjs", ".cjs"}),
    Language(tree_sitter_javascript.language()),
)

SPECS: tuple[LanguageSpec, ...] = (PYTHON, TYPESCRIPT, TSX, JAVASCRIPT)


def spec_for(path: str) -> LanguageSpec | None:
    """Return the spec covering ``path``'s extension, or ``None`` if unsupported."""
    suffix = posixpath.splitext(path)[1]
    for spec in SPECS:
        if suffix in spec.extensions:
            return spec
    return None
