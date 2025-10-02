import asyncio
import json
from typing import AsyncGenerator, Iterable, List, Optional

import httpx


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        embed_model: str,
        timeout: float = 120.0,
        temperature: float = 0.2,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._embed_model = embed_model
        self._timeout = timeout
        self._temperature = temperature
        self._async_client: Optional[httpx.AsyncClient] = None
        self._sync_client: Optional[httpx.Client] = None
        self._client_lock = asyncio.Lock()

    async def _get_async_client(self) -> httpx.AsyncClient:
        async with self._client_lock:
            if self._async_client is None:
                self._async_client = httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=self._timeout,
                )
        return self._async_client

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._sync_client

    @staticmethod
    def _extract_embedding(data: dict) -> List[float]:
        embedding = data.get("embedding")
        if embedding is None:
            embedding = data.get("embeddings")
        if embedding is None:
            raise ValueError("embedding key missing in response")
        if embedding and isinstance(embedding[0], list):
            if len(embedding) != 1:
                raise ValueError("expected single embedding vector")
            embedding = embedding[0]
        if not isinstance(embedding, list):
            raise ValueError("embedding payload is not a list")
        return embedding

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self._temperature,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        if context:
            payload["context"] = context

        client = await self._get_async_client()
        async with client.stream("POST", "/api/generate", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("done"):
                    break
                chunk = data.get("response")
                if chunk:
                    yield chunk

    async def embed(self, texts: Iterable[str]) -> List[List[float]]:
        client = await self._get_async_client()
        results: List[List[float]] = []
        for text in texts:
            payload = {
                "model": self._embed_model,
                "prompt": text,
            }
            response = await client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            vector = self._extract_embedding(response.json())
            results.append(vector)
        return results

    def embed_sync(self, texts: Iterable[str]) -> List[List[float]]:
        client = self._get_sync_client()
        results: List[List[float]] = []
        for text in texts:
            payload = {
                "model": self._embed_model,
                "prompt": text,
            }
            response = client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            vector = self._extract_embedding(response.json())
            results.append(vector)
        return results

    async def aclose(self) -> None:
        async with self._client_lock:
            if self._async_client is not None:
                await self._async_client.aclose()
                self._async_client = None
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None


__all__ = ["OllamaClient"]
