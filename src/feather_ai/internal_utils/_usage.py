"""
Helpers for extracting LLM token usage from LangChain messages / chunks.

Usage information lives on ``AIMessage.usage_metadata`` (and on aggregated
``AIMessageChunk`` objects once the stream has been summed together). These
helpers turn that into a :class:`~feather_ai.types.response.UsageInfo` object.

The extraction is intentionally defensive: if usage metadata is missing (some
providers/paths do not attach it, e.g. structured-output runnables that return a
parsed pydantic object) we return a ``UsageInfo`` with zero counts rather than
raising. Capturing usage must never break a generation.
"""
from typing import Any, Optional

from ..types.response import UsageInfo


def _model_name(message: Any, llm: Any = None) -> str:
    """
    Best-effort resolution of the model name.

    Tries the message ``response_metadata`` first (most accurate for the actual
    call), then falls back to the ``model`` / ``model_name`` attribute of the llm
    object, unwrapping a bound Runnable if necessary. Provider prefixes such as
    ``models/`` are trivially stripped.
    """
    name: Optional[str] = None

    response_metadata = getattr(message, "response_metadata", None) or {}
    if isinstance(response_metadata, dict):
        name = (
            response_metadata.get("model_name")
            or response_metadata.get("model")
            or response_metadata.get("model_id")
        )

    if not name and llm is not None:
        name = getattr(llm, "model", None) or getattr(llm, "model_name", None)
        if not name:
            # llm may be a RunnableBinding (bound tools / structured output)
            bound = getattr(llm, "bound", None)
            if bound is not None:
                name = getattr(bound, "model", None) or getattr(bound, "model_name", None)

    if not name:
        return ""

    name = str(name)
    if "/" in name:
        name = name.split("/")[-1]
    return name


def extract_usage(message: Any, llm: Any = None) -> UsageInfo:
    """
    Build a :class:`UsageInfo` (with ``calls=1``) from a completed LLM message.

    Args:
        message: an ``AIMessage`` or aggregated ``AIMessageChunk`` (or any object
            that may carry ``usage_metadata``). May also be a parsed pydantic
            object from a structured-output runnable, in which case zeros are
            returned.
        llm: optional llm object used to resolve the model name as a fallback.

    Returns:
        A ``UsageInfo`` for exactly one LLM invocation. Never raises.
    """
    model = ""
    try:
        model = _model_name(message, llm)
    except Exception:
        model = ""

    usage_metadata = getattr(message, "usage_metadata", None)
    if not usage_metadata or not isinstance(usage_metadata, dict):
        # No usage available for this call; still count it as one invocation.
        return UsageInfo(model=model, calls=1)

    input_tokens = usage_metadata.get("input_tokens", 0) or 0
    output_tokens = usage_metadata.get("output_tokens", 0) or 0
    total_tokens = usage_metadata.get("total_tokens", 0) or 0

    input_token_details = usage_metadata.get("input_token_details", {}) or {}
    cache_read_tokens = 0
    if isinstance(input_token_details, dict):
        cache_read_tokens = input_token_details.get("cache_read", 0) or 0

    return UsageInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_read_tokens=cache_read_tokens,
        model=model,
        calls=1,
    )
