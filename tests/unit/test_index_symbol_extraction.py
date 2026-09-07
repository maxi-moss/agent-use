"""extract_file over the checked-in fixture repos: exact symbols, signatures,
references and imports per language."""

from pathlib import Path

from broker.index.languages import JAVASCRIPT, PYTHON, TYPESCRIPT, LanguageSpec
from broker.index.schemas import FileExtraction, Import, RefKind, Symbol, SymbolKind
from broker.index.symbol_extraction import extract_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "indexer_inputs"
PY_REPO = FIXTURES / "python_repo"
TS_REPO = FIXTURES / "typescript_repo"


def extract(repo: Path, rel: str, spec: LanguageSpec) -> FileExtraction:
    return extract_file(spec, rel, (repo / rel).read_bytes())


def by_name(extraction: FileExtraction) -> dict[str, Symbol]:
    return {s.qualified_name: s for s in extraction.symbols}


def refs(extraction: FileExtraction, source: str) -> set[tuple[RefKind, str]]:
    return {(r.kind, r.target) for r in extraction.references if r.source == source}


# ── Python ────────────────────────────────────────────────────────────────


def test_python_symbols_signatures_and_nesting() -> None:
    extraction = extract(PY_REPO, "app/chat.py", PYTHON)
    symbols = by_name(extraction)
    assert set(symbols) == {
        "app/chat.py::ChatService",
        "app/chat.py::ChatService.__init__",
        "app/chat.py::ChatService.send",
        "app/chat.py::ChatService._prepare",
        "app/chat.py::render",
    }
    cls = symbols["app/chat.py::ChatService"]
    assert cls.kind is SymbolKind.CLASS
    assert cls.signature == "class ChatService:"
    assert cls.docstring == "Sends messages through the configured provider."
    send = symbols["app/chat.py::ChatService.send"]
    assert send.kind is SymbolKind.METHOD
    assert send.signature == "def send(self, message: str) -> str:"
    assert send.body.startswith("return self.provider.complete(")
    # the nested `strip` is not a symbol; its call belongs to `_prepare`.
    # Annotation names are collected as written (`str` included); resolution
    # drops the ones that name nothing in the repository.
    assert refs(extraction, "app/chat.py::ChatService._prepare") == {
        (RefKind.CALL, "strip"),
        (RefKind.CALL, "text.strip"),
        (RefKind.TYPE, "str"),
    }
    assert refs(extraction, "app/chat.py::ChatService.send") == {
        (RefKind.CALL, "self.provider.complete"),
        (RefKind.CALL, "self._prepare"),
        (RefKind.TYPE, "str"),
    }
    assert refs(extraction, "app/chat.py::ChatService.__init__") == {
        (RefKind.CALL, "factory.create_provider"),
        (RefKind.TYPE, "Settings"),
    }


def test_python_overloads_collapse_into_one_symbol() -> None:
    render = by_name(extract(PY_REPO, "app/chat.py", PYTHON))["app/chat.py::render"]
    assert render.kind is SymbolKind.FUNCTION
    assert render.signature == (
        "@overload\ndef render(value: str) -> str:\n"
        "@overload\ndef render(value: int) -> str:\n"
        "def render(value: str | int) -> str:"
    )
    assert render.docstring == "Render a value."
    assert "return str(value)" in render.body


def test_python_fields_and_imports() -> None:
    extraction = extract(PY_REPO, "app/settings.py", PYTHON)
    symbols = by_name(extraction)
    assert symbols["app/settings.py::Settings"].fields == [
        "provider: str",
        "anthropic_api_key: str",
    ]
    assert extraction.imports == [
        Import(
            path="app/settings.py",
            local_name="BaseModel",
            module="pydantic",
            imported_name="BaseModel",
        )
    ]
    assert refs(extraction, "app/settings.py::Settings") == {
        (RefKind.BASE, "BaseModel"),
        (RefKind.TYPE, "str"),  # from the field annotations, attributed to the class
    }


def test_python_relative_and_module_imports() -> None:
    anthropic = extract(PY_REPO, "app/providers/anthropic.py", PYTHON)
    assert anthropic.imports == [
        Import(
            path="app/providers/anthropic.py",
            local_name="Provider",
            module="app.providers.base",
            imported_name="Provider",
        )
    ]
    chat = extract(PY_REPO, "app/chat.py", PYTHON)
    assert Import(
        path="app/chat.py",
        local_name="factory",
        module="app.providers",
        imported_name="factory",
    ) in chat.imports


def test_python_function_references() -> None:
    extraction = extract(PY_REPO, "app/providers/factory.py", PYTHON)
    assert refs(extraction, "app/providers/factory.py::create_provider") == {
        (RefKind.CALL, "AnthropicProvider"),
        (RefKind.CALL, "ValueError"),
        (RefKind.TYPE, "Settings"),
        (RefKind.TYPE, "Provider"),
    }


# ── TypeScript / JavaScript ───────────────────────────────────────────────


def test_typescript_class_arrow_method_and_module_arrow() -> None:
    extraction = extract(TS_REPO, "src/chat.ts", TYPESCRIPT)
    symbols = by_name(extraction)
    assert set(symbols) == {
        "src/chat.ts::ChatService",
        "src/chat.ts::ChatService.constructor",
        "src/chat.ts::ChatService.send",
        "src/chat.ts::handleChat",
    }
    cls = symbols["src/chat.ts::ChatService"]
    assert cls.signature == "export class ChatService"
    assert cls.docstring == "/** Sends messages through the configured provider. */"
    assert cls.fields == ["provider: Provider"]
    send = symbols["src/chat.ts::ChatService.send"]
    assert send.kind is SymbolKind.METHOD
    assert send.signature == "send = async (message: string): Promise<string> =>"
    handle = symbols["src/chat.ts::handleChat"]
    assert handle.kind is SymbolKind.FUNCTION
    assert handle.signature == (
        "export const handleChat = async (service: ChatService, message: string) =>"
    )
    assert refs(extraction, "src/chat.ts::ChatService.constructor") == {
        (RefKind.CALL, "createProvider"),
        (RefKind.TYPE, "ProviderSettings"),
    }
    assert refs(extraction, "src/chat.ts::ChatService") == {(RefKind.TYPE, "Provider")}
    assert refs(extraction, "src/chat.ts::handleChat") == {
        (RefKind.CALL, "service.send"),
        (RefKind.TYPE, "ChatService"),
    }
    assert Import(
        path="src/chat.ts",
        local_name="createProvider",
        module="src/providers/factory",
        imported_name="createProvider",
    ) in extraction.imports


def test_typescript_types_and_abstract_members() -> None:
    settings = by_name(extract(TS_REPO, "src/settings.ts", TYPESCRIPT))
    assert settings["src/settings.ts::ProviderSettings"].kind is SymbolKind.TYPE
    assert settings["src/settings.ts::ProviderSettings"].signature == (
        "export interface ProviderSettings"
    )
    base = extract(TS_REPO, "src/providers/base.ts", TYPESCRIPT)
    complete = by_name(base)["src/providers/base.ts::Provider.complete"]
    assert complete.signature == "abstract complete(prompt: string): Promise<string>"
    assert complete.body == ""
    anthropic = extract(TS_REPO, "src/providers/anthropic.ts", TYPESCRIPT)
    assert refs(anthropic, "src/providers/anthropic.ts::AnthropicProvider") == {
        (RefKind.BASE, "Provider"),
    }
    assert refs(anthropic, "src/providers/anthropic.ts::AnthropicProvider.complete") == {
        (RefKind.CALL, "this.call"),
        (RefKind.TYPE, "Promise"),  # generic_type → type_identifier; `string` is predefined_type
    }
    factory = extract(TS_REPO, "src/providers/factory.ts", TYPESCRIPT)
    assert (RefKind.CALL, "AnthropicProvider") in refs(
        factory, "src/providers/factory.ts::createProvider"
    )


def test_javascript_function_and_import() -> None:
    extraction = extract(TS_REPO, "src/index.js", JAVASCRIPT)
    main = by_name(extraction)["src/index.js::main"]
    assert main.signature == "export function main(service)"
    assert refs(extraction, "src/index.js::main") == {(RefKind.CALL, "handleChat")}
    assert extraction.imports == [
        Import(
            path="src/index.js",
            local_name="handleChat",
            module="src/chat",
            imported_name="handleChat",
        )
    ]
