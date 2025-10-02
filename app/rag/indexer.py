import argparse
import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from app.config import Settings, get_settings
from app.ollama_client import OllamaClient

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".rst"}


def iter_docs(docs_dir: Path) -> Iterable[Path]:
    for path in docs_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def load_text(path: Path) -> str:
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
    return chunks


def compute_docs_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
    return digest.hexdigest()


def build_index(settings: Settings, client: OllamaClient) -> None:
    docs = list(iter_docs(settings.docs_dir))
    if not docs:
        raise RuntimeError(f"no supported documents found in {settings.docs_dir}")

    all_chunks: List[str] = []
    metadata: List[dict] = []

    for doc_path in docs:
        text = load_text(doc_path)
        if not text:
            continue
        for chunk_index, (offset, chunk) in enumerate(chunk_text(text, settings.chunk_size, settings.chunk_overlap)):
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

    embeddings: List[List[float]] = []
    batch_size = 32
    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start : start + batch_size]
        vectors = client.embed_sync(batch)
        if len(vectors) != len(batch):
            raise RuntimeError("embedding response length mismatch")
        embeddings.extend(vectors)

    emb_array = np.asarray(embeddings, dtype=np.float32)
    index_path = settings.index_dir / "embeddings.npy"
    meta_path = settings.index_dir / "meta.json"
    stats_path = settings.index_dir / "stats.json"

    np.save(index_path, emb_array)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {
        "model": settings.embed_model,
        "docs_hash": compute_docs_hash(docs),
        "chunk_count": len(all_chunks),
        "built_at": datetime.utcnow().isoformat() + "Z",
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


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
