from __future__ import annotations

import logging
from typing import Any, Dict, Annotated

from agents import function_tool
from pydantic import Field

from app.web_search_client import WebSearchClient

LOGGER = logging.getLogger(__name__)

QueryArg = Annotated[str, Field(description="Search query string")]
MaxResultsArg = Annotated[
    int | None,
    Field(ge=1, le=10, description="Maximum number of search results to return (1-10)"),
]


def _clamp_max_results(value: int | None, default: int) -> int:
    """Return a value between 1 and 10 inclusive."""
    effective = value or default
    return max(1, min(10, effective))


def create_disabled_web_search_tool(message: str | None = None):
    """Return a placeholder tool used when web search is disabled."""
    notice = message or "web_search is disabled for this deployment."

    @function_tool(name_override="web_search")
    async def web_search(query: QueryArg, max_results: MaxResultsArg = None) -> Dict[str, Any]:
        """Inform callers that web search is disabled."""
        LOGGER.exception("web_search requested while disabled: %s", query)
        return {"results": [], "error": notice, "query": query}

    return web_search


def create_ollama_web_search_tool(
    client: WebSearchClient,
    default_max_results: int = 5,
):
    """Return an Agents tool that calls the Ollama Web Search API."""

    @function_tool(name_override="web_search")
    async def web_search(query: QueryArg, max_results: MaxResultsArg = None) -> Dict[str, Any]:
        """Search the web via the Ollama Web Search API."""
        effective_max = _clamp_max_results(max_results, default_max_results)
        try:
            results = await client.search(query, effective_max)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("web_search tool failed")
            return {
                "results": [],
                "error": f"web_search failed: {exc}",
                "query": query,
            }
        return {"results": results, "query": query}

    return web_search


__all__ = ["create_ollama_web_search_tool", "create_disabled_web_search_tool"]
