"""Grounding: propose a session's initial task prompt from the intent and its code.

Class names, field names, `Field` descriptions and docstrings of these models
are sent to the model.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from anthropic.types import MessageParam, TextBlockParam, ToolParam
from pydantic import BaseModel, ConfigDict

from broker import llm_timing
from broker import prompts
from broker.config import EmbeddingConfig, SessionModelConfig
from broker.index.embedding import OpenAIEmbedder
from broker.index.render import fit_to_budget, render_relevant_code
from broker.index.retrieval import retrieve as retrieve_code
from broker.index.schemas import GroundingContext
from broker.llm import LLMCaller, ToolCall, strict_tool
from broker.paths import BrokerPaths
from broker.session.llm_stack import forced_call

logger = logging.getLogger(__name__)


class ProposePromptCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    prompt: str


GROUNDING_TOOLS: list[ToolParam] = [
    strict_tool(
        "propose_prompt",
        "Propose the initial task prompt for the coding session. The developer"
        " reviews and may revise it before submission.",
        ProposePromptCall,
    ),
]

_TOOL_MODELS: dict[str, type[ProposePromptCall]] = {"propose_prompt": ProposePromptCall}

_GROUNDING_PROMPT = prompts.load("grounding")


class Retriever(Protocol):
    """The injected retrieval seam: tests pass a fake, production binds the index."""

    async def __call__(self, intent: str, cwd: Path) -> GroundingContext:
        """Return the intent's code neighbourhood for the repository at ``cwd``."""
        ...


@dataclass
class Grounding:
    """A proposed prompt with the code neighbourhood it was grounded in."""

    proposal: ProposePromptCall
    context: GroundingContext


def bind_index_retriever(paths: BrokerPaths, embedding: EmbeddingConfig) -> Retriever:
    """Bind index retrieval to this broker home and the pinned embedding model.

    The embedder is built per call so a missing ``OPENAI_API_KEY`` fails at
    grounding time, loudly, rather than at broker start.

    Args:
        paths: Resolves the repository's index file.
        embedding: The model pinned in ``broker.config``, as sent by the master.

    Returns:
        A callable matching ``Retriever``.
    """

    async def call(intent: str, cwd: Path) -> GroundingContext:
        repo = cwd.resolve()
        return await retrieve_code(
            intent,
            repo,
            index_path=paths.index_db(repo),
            embedder=OpenAIEmbedder.from_env(embedding),
        )

    return call


@llm_timing.timed("retrieval")
async def _retrieve(retrieve: Retriever, intent: str, cwd: Path) -> GroundingContext:
    """Retrieve and trim the neighbourhood so the rendered block fits its budget."""
    return fit_to_budget(await retrieve(intent, cwd))


@llm_timing.timed("grounding")
async def _propose(
    llm_call: LLMCaller[ToolCall], model_cfg: SessionModelConfig, parts: list[str]
) -> ProposePromptCall:
    """Make the one grounding call and validate its forced tool use.

    Raises:
        LLMCallError: The LLM called a tool other than ``propose_prompt``, or
            the tool input failed validation. Uncaught here; the caller
            aborts the spawn as a fatal session error.
    """
    system: list[TextBlockParam] = [{"type": "text", "text": _GROUNDING_PROMPT}]
    messages: list[MessageParam] = [{"role": "user", "content": "\n\n".join(parts)}]
    return await forced_call(
        llm_call,
        model_cfg,
        system=system,
        messages=messages,
        tools=GROUNDING_TOOLS,
        models=_TOOL_MODELS,
        label="grounding",
    )


async def ground_intent(
    llm_call: LLMCaller[ToolCall],
    model_cfg: SessionModelConfig,
    *,
    retrieve: Retriever,
    intent: str,
    cwd: Path,
) -> Grounding:
    """Propose the initial task prompt for a session from the stated intent.

    Args:
        llm_call: The injected tool-calling seam.
        model_cfg: Supplies the model id and the token cap.
        retrieve: The injected code-retrieval seam.
        intent: The developer's intent, passed verbatim.
        cwd: Session working directory: the repository root.

    Returns:
        The validated ``propose_prompt`` call and the neighbourhood it saw.

    Raises:
        LLMCallError: The grounding call failed or returned the wrong tool.
        Exception: Whatever ``retrieve`` raises — retrieval failures abort
            grounding; there is no degraded path. Neither is caught here;
            the caller aborts the spawn as a fatal session error.
    """
    context = await _retrieve(retrieve, intent, cwd)
    relevant_code = render_relevant_code(context)
    logger.info("relevant code for grounding in %s:\n%s", cwd, relevant_code)
    claude_md = ""
    claude_md_path = cwd / "CLAUDE.md"
    if claude_md_path.exists():
        claude_md = claude_md_path.read_text(encoding="utf-8")
    parts = [f"# Developer intent (verbatim)\n{intent}"]
    if claude_md:
        parts.append(f"# The codebase's CLAUDE.md\n{claude_md}")
    parts.append(relevant_code)
    proposal = await _propose(llm_call, model_cfg, parts)
    return Grounding(proposal=proposal, context=context)
