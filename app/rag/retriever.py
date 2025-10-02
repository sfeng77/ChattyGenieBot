import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from app.ollama_client import OllamaClient

LOGGER = logging.getLogger(__name__)


@dataclass
class Retrieval:
    text: str
    doc_path: str
    chunk_index: int
    offset: int
    score: float


class VectorIndex:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self._embeddings: np.ndarray | None = None
        self._normalized: np.ndarray | None = None
        self._metadata: List[dict] = []

    def load(self) -> None:
        emb_path = self.index_dir / "embeddings.npy"
        meta_path = self.index_dir / "meta.json"
        LOGGER.debug("Loading embeddings from %s", emb_path)
        LOGGER.debug("Loading metadata from %s", meta_path)
        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError("RAG index not found; run the indexer first")
        self._embeddings = np.load(emb_path)
        with meta_path.open("r", encoding="utf-8") as fh:
            self._metadata = json.load(fh)
        if len(self._metadata) != len(self._embeddings):
            raise ValueError("metadata and embeddings length mismatch")
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        self._normalized = self._embeddings / norms
        LOGGER.info("Vector index ready (chunks=%s, dim=%s)", len(self._metadata), self._normalized.shape[1])

    def is_ready(self) -> bool:
        ready = self._normalized is not None and len(self._metadata) > 0
        LOGGER.debug("Index readiness checked: %s", ready)
        return ready

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Retrieval]:
        if self._normalized is None:
            raise RuntimeError("index not loaded")
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            LOGGER.debug("Query vector norm is zero; returning no results")
            return []
        query_normalized = query_vec / query_norm
        scores = query_normalized @ self._normalized.T
        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[Retrieval] = []
        for idx in top_indices:
            meta = self._metadata[int(idx)]
            score = float(scores[int(idx)])
            LOGGER.debug(
                "Hit chunk idx=%s score=%.4f doc=%s#%s",
                int(idx),
                score,
                meta.get("doc_path"),
                meta.get("chunk_index"),
            )
            results.append(
                Retrieval(
                    text=meta["text"],
                    doc_path=meta["doc_path"],
                    chunk_index=int(meta["chunk_index"]),
                    offset=int(meta["offset"]),
                    score=score,
                )
            )
        return results


class RAGRetriever:
    def __init__(self, index_dir: Path, top_k: int, client: OllamaClient) -> None:
        self._index = VectorIndex(index_dir)
        self._top_k = top_k
        self._client = client
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            LOGGER.debug("Lazy-loading vector index")
            self._index.load()
            self._loaded = True

    async def retrieve(self, query: str) -> List[Retrieval]:
        try:
            self._ensure_loaded()
        except FileNotFoundError:
            LOGGER.warning("Vector index missing; returning empty retrievals")
            return []
        LOGGER.debug("Retrieving for query chars=%s", len(query))
        vectors = await self._client.embed([query])
        if not vectors:
            LOGGER.debug("Embedding call returned no vectors")
            return []
        query_vec = np.asarray(vectors[0], dtype=np.float32)
        results = self._index.search(query_vec, self._top_k)
        LOGGER.debug("Returning %s retrievals", len(results))
        return results


__all__ = ["RAGRetriever", "Retrieval", "VectorIndex"]
