from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable

from pinecone import Pinecone


def upsert_vectors(
    vectors: list[tuple[str, list[float], dict]],
    batch_size: int = 100,
    namespace: str | None = None,
) -> None:
    index = _get_index()
    namespace = _resolve_namespace(namespace)
    for batch in _batch(vectors, batch_size):
        index.upsert(vectors=batch, namespace=namespace)


def query_vectors(
    vector: list[float],
    top_k: int,
    filters: dict | None = None,
    namespace: str | None = None,
) -> list[dict]:
    index = _get_index()
    namespace = _resolve_namespace(namespace)
    response = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=filters,
        namespace=namespace,
    )
    return response.get("matches", [])


@lru_cache(maxsize=1)
def _get_index():
    api_key = os.getenv("PINECONE_API_KEY", "")
    index_name = os.getenv("PINECONE_INDEX", "")
    if not api_key or not index_name:
        raise ValueError("PINECONE_API_KEY and PINECONE_INDEX must be set.")

    client = Pinecone(api_key=api_key)
    indexes = {index["name"] for index in client.list_indexes()}
    if index_name not in indexes:
        raise ValueError(
            f"Pinecone index '{index_name}' not found. Create it in Pinecone before running."
        )
    return client.Index(index_name)


def _resolve_namespace(namespace: str | None) -> str | None:
    return namespace or os.getenv("PINECONE_NAMESPACE")


def _batch(items: list[tuple[str, list[float], dict]], size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
