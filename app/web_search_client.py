from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

LOGGER = logging.getLogger(__name__)


class WebSearchClient:
    """Client for the Ollama Web Search API."""

    def __init__(self, base_url: str, endpoint: str, timeout: float, api_key: str | None) -> None:
        self._base_url = base_url.rstrip("/")
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        self._endpoint = endpoint
        self._timeout = timeout
        self._api_key = api_key

    async def search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        if not self._api_key:
            raise ValueError("OLLAMA_API_KEY must be configured to use web_search")
        payload = {
            "query": query,
            "max_results": max_results,
        }
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
        }
        url = f"{self._base_url}{self._endpoint}"
        LOGGER.debug(
            "Calling Ollama web search (url=%s, query=%s, max_results=%s)",
            url,
            query,
            max_results,
        )
        async with httpx.AsyncClient(timeout=self._timeout, headers=headers) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        results = data.get("results") or []
        normalized: List[Dict[str, str]] = []
        for item in results:
            title = (item.get("title") or "").strip()
            url_value = (item.get("url") or "").strip()
            snippet = (item.get("content") or "").strip()
            if len(snippet) > 250:
                snippet = f"{snippet[:247]}..."
            normalized.append(
                {
                    "title": title or url_value or "Untitled result",
                    "url": url_value,
                    "snippet": snippet,
                }
            )
        LOGGER.debug("Web search returned %s results", len(normalized))
        return normalized


__all__ = ["WebSearchClient"]
