"""Prompt loading. Prompts are STATIC prefixes — no .format(), no f-strings,
no timestamps (cache determinism). Volatile content goes after the
cache breakpoint, never into these files."""

from importlib import resources


def load(name: str) -> str:
    """Read the static prompt file ``<name>.md`` from this package.

    Args:
        name: Prompt basename, without the ``.md`` suffix.

    Returns:
        The file's text verbatim — never formatted or interpolated.

    Raises:
        ValueError: The prompt file is empty or holds only whitespace.
    """
    text = (
        resources.files("broker.prompts")
        .joinpath(f"{name}.md")
        .read_text(encoding="utf-8")
    )
    if not text.strip():
        raise ValueError(f"prompt {name!r} is empty")
    return text
