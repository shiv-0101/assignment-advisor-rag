from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Iterable

import requests


def embed_texts(texts: list[str], batch_size: int = 8, max_retries: int = 2) -> list[list[float]]:
    provider = os.getenv("EMBEDDING_PROVIDER", "hf").lower()
    if provider == "local":
        return _embed_texts_local(texts)
    if provider != "hf":
        raise ValueError("EMBEDDING_PROVIDER must be 'hf' or 'local'.")

    api_key = os.getenv("HF_API_KEY", "")
    model = os.getenv("HF_EMBEDDING_MODEL", "")
    if not api_key or not model:
        raise ValueError("HF_API_KEY and HF_EMBEDDING_MODEL must be set.")

    urls = [
        f"https://router.huggingface.co/hf-inference/pipeline/feature-extraction/{model}",
        f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}",
        f"https://router.huggingface.co/hf-inference/models/{model}",
        f"https://api-inference.huggingface.co/models/{model}",
    ]
    headers = {"Authorization": f"Bearer {api_key}"}

    embeddings: list[list[float]] = []
    for batch in _batch(texts, batch_size):
        data = None
        last_error = None
        for url in urls:
            for attempt in range(max_retries + 1):
                response = requests.post(
                    url,
                    headers=headers,
                    json={"inputs": batch, "options": {"wait_for_model": True}},
                    timeout=60,
                )
                if response.status_code in {429, 503} and attempt < max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if response.status_code == 400 and "Model not supported" in response.text:
                    raise RuntimeError(
                        "Hugging Face API error: model not supported on hf-inference. "
                        "Set EMBEDDING_PROVIDER=local or use a paid endpoint."
                    )
                if response.status_code == 404:
                    last_error = response.text
                    break
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Hugging Face API error: {response.status_code} {response.text}"
                    )
                data = response.json()
                break
            if data is not None or response.status_code != 404:
                break

        if data is None:
            raise RuntimeError(
                "Hugging Face API error: 404 for all endpoints. "
                "Check model name and token, then retry."
            )
        embeddings.extend(_pool_embeddings(data))
    return embeddings


def _embed_texts_local(texts: list[str]) -> list[list[float]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Local embeddings require sentence-transformers. Install it and retry."
        ) from exc

    model_name = os.getenv("EMBEDDING_LOCAL_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    model = _get_local_model(model_name, SentenceTransformer)
    vectors = model.encode(texts, normalize_embeddings=True).tolist()
    return [list(map(float, vector)) for vector in vectors]


@lru_cache(maxsize=1)
def _get_local_model(model_name: str, loader) -> object:
    return loader(model_name)


def _batch(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _pool_embeddings(data: list) -> list[list[float]]:
    if not data:
        return []

    if isinstance(data[0], (int, float)):
        return [data]

    if isinstance(data[0], list) and data and isinstance(data[0][0], (int, float)):
        return [_mean_pool(data)]

    pooled: list[list[float]] = []
    for item in data:
        if not item:
            pooled.append([])
            continue
        if isinstance(item[0], (int, float)):
            pooled.append(item)
        else:
            pooled.append(_mean_pool(item))
    return pooled


def _mean_pool(token_vectors: list[list[float]]) -> list[float]:
    length = len(token_vectors)
    if length == 0:
        return []
    dims = len(token_vectors[0])
    sums = [0.0] * dims
    for vector in token_vectors:
        for i, value in enumerate(vector):
            sums[i] += float(value)
    return [value / length for value in sums]
