import argparse
import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from app.config import Settings, get_settings
from app.logging_utils import setup_logging
from app.ollama_client import OllamaClient

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".rst"}

LOGGER = logging.getLogger(__name__)


def iter_docs(docs_dir: Path) -> Iterable[Path]:
    for path in docs_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            LOGGER.debug("Found document: %s", path)
            yield path
        else:
            LOGGER.debug("Ignoring unsupported file: %s", path)


def load_text(path: Path) -> str:
    LOGGER.debug("Loading text from %s", path)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int, overlap: int) -> Sequence[Tuple[int, str]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    overlap = max(0, min(overlap, size - 1))
    chunks: List[Tuple[int, str]] = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, chunk))
        if end >= len(text):
            break
        start = end - overlap
    LOGGER.debug("Created %s chunks (size=%s, overlap=%s)", len(chunks), size, overlap)
    return chunks


def compute_docs_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
    hexdigest = digest.hexdigest()
    LOGGER.debug("Computed docs hash: %s", hexdigest)
    return hexdigest


def build_index(settings: Settings, client: OllamaClient) -> None:
    LOGGER.debug(
        "Building index (docs_dir=%s, index_dir=%s, chunk_size=%s, overlap=%s)",
        settings.docs_dir,
        settings.index_dir,
        settings.chunk_size,
        settings.chunk_overlap,
    )
    docs = list(iter_docs(settings.docs_dir))
    if not docs:
        raise RuntimeError(f"no supported documents found in {settings.docs_dir}")

    LOGGER.info("Indexing %s documents", len(docs))

    all_chunks: List[str] = []
    metadata: List[dict] = []

    for doc_path in docs:
        text = load_text(doc_path)
        if not text:
            LOGGER.debug("Skipping empty document: %s", doc_path)
            continue
        doc_chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        LOGGER.debug("Document %s produced %s chunks", doc_path, len(doc_chunks))
        for chunk_index, (offset, chunk) in enumerate(doc_chunks):
            all_chunks.append(chunk)
            metadata.append(
                {
                    "doc_path": str(doc_path),
                    "chunk_index": chunk_index,
                    "offset": offset,
                    "text": chunk,
                }
            )

    if not all_chunks:
        raise RuntimeError("no text chunks produced; check chunk settings")

    LOGGER.info("Embedding %s chunks", len(all_chunks))

    embeddings: List[List[float]] = []
    batch_size = 32
    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start : start + batch_size]
        LOGGER.debug("Embedding batch start=%s size=%s", start, len(batch))
        vectors = client.embed_sync(batch)
        if len(vectors) != len(batch):
            raise RuntimeError("embedding response length mismatch")
        embeddings.extend(vectors)

    emb_array = np.asarray(embeddings, dtype=np.float32)
    index_path = settings.index_dir / "embeddings.npy"
    meta_path = settings.index_dir / "meta.json"
    stats_path = settings.index_dir / "stats.json"

    LOGGER.debug("Writing embeddings to %s", index_path)
    np.save(index_path, emb_array)
    LOGGER.debug("Writing metadata to %s", meta_path)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {
        "model": settings.embed_model,
        "docs_hash": compute_docs_hash(docs),
        "chunk_count": len(all_chunks),
        "built_at": datetime.utcnow().isoformat() + "Z",
    }
    LOGGER.debug("Writing stats to %s", stats_path)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    LOGGER.info("Index build complete (chunks=%s, dim=%s)", emb_array.shape[0], emb_array.shape[1])


async def build_index_async(settings: Settings, client: OllamaClient) -> None:
    await asyncio.to_thread(build_index, settings, client)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG index using Ollama embeddings")
    parser.add_argument("--docs", type=Path, help="Docs directory", default=None)
    parser.add_argument("--index", type=Path, help="Index directory", default=None)
    args = parser.parse_args()

    settings = get_settings()
    if args.docs is not None:
        settings.docs_dir = args.docs
    if args.index is not None:
        settings.index_dir = args.index

    setup_logging(settings)

    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
        embed_model=settings.embed_model,
        timeout=settings.ollama_request_timeout,
        temperature=settings.ollama_temperature,
    )
    try:
        build_index(settings, client)
    finally:
        asyncio.run(client.aclose())


if __name__ == "__main__":
    main()
