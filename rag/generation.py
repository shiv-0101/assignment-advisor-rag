from __future__ import annotations

import os

import requests


def generate_feedback(question: str, guidelines_chunks: list[str], draft_chunks: list[str]) -> str:
    api_key = os.getenv("GROQ_API_KEY", "")
    model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    if not api_key:
        raise ValueError("GROQ_API_KEY must be set.")

    guidelines_text = "\n\n".join(guidelines_chunks)
    draft_text = "\n\n".join(draft_chunks)

    prompt = (
        "You are an academic writing assistant.\n"
        "Use ONLY the provided Guidelines and Draft excerpts.\n"
        "Identify missing points, mismatches, or improvements as bullet points.\n"
        "If everything matches, say 'No missing points found' and add 1 small improvement.\n\n"
        f"Question: {question}\n\n"
        "Guidelines excerpts:\n"
        f"{guidelines_text}\n\n"
        "Draft excerpts:\n"
        f"{draft_text}\n"
    )

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "Be concise and structured."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Groq API error: {response.status_code} {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()
