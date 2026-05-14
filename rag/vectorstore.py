from __future__ import annotations


def upsert_vectors(vectors: list[tuple[str, list[float], dict]]) -> None:
    raise NotImplementedError("Pinecone upsert will be implemented in Phase 3.")


def query_vectors(vector: list[float], top_k: int, filters: dict | None = None) -> list[dict]:
    raise NotImplementedError("Pinecone query will be implemented in Phase 4.")
