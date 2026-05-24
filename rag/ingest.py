from __future__ import annotations

import hashlib
import io
import re
from typing import Iterable

import pdfplumber
from docx import Document


HEADING_REGEX = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?([A-Z][A-Z\s\-]{2,}|[A-Z][\w\s\-]{1,})$"
)
SECTION_KEYWORDS = {
    "conclusion",
    "requirements",
    "rubric",
    "grading",
    "introduction",
    "abstract",
    "references",
    "bibliography",
    "method",
    "methods",
    "results",
    "discussion",
}


def load_pdf_text(file_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


def load_docx_text(file_bytes: bytes) -> str:
    document = Document(io.BytesIO(file_bytes))
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    return "\n".join(paragraphs)


def normalize_text(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    doc_type: str,
    source_file: str,
    chunk_size: int = 2200,
    overlap: int = 250,
) -> list[dict]:
    sections = _split_by_headings(text)
    chunks: list[dict] = []
    for section, section_text in sections:
        for chunk_index, chunk in enumerate(_chunk_by_size(section_text, chunk_size, overlap)):
            chunk_id = _make_chunk_id(doc_type, section, chunk_index, chunk)
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_type": doc_type,
                    "section": section,
                    "source_file": source_file,
                    "text": chunk,
                }
            )
    return chunks


def detect_headings(text: str) -> list[str]:
    headings: list[str] = []
    for line in _iter_lines(text):
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) > 80:
            continue
        match = HEADING_REGEX.match(stripped)
        if not match:
            continue
        normalized = match.group(1).strip().lower()
        if normalized in SECTION_KEYWORDS and normalized not in headings:
            headings.append(normalized)
    return headings


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    current_section = "unknown"
    buffer: list[str] = []
    sections: list[tuple[str, str]] = []

    for line in _iter_lines(text):
        stripped = line.strip()
        if _is_heading_line(stripped):
            if buffer:
                sections.append((current_section, "\n".join(buffer).strip()))
                buffer = []
            current_section = stripped.lower()
            continue
        buffer.append(line)

    if buffer:
        sections.append((current_section, "\n".join(buffer).strip()))

    return [(section, body) for section, body in sections if body]


def _is_heading_line(line: str) -> bool:
    if not line or len(line) > 80:
        return False
    match = HEADING_REGEX.match(line)
    if not match:
        return False
    normalized = match.group(1).strip().lower()
    return normalized in SECTION_KEYWORDS


def _chunk_by_size(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        return []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    chunks: list[str] = []
    start = 0
    length = len(text)
    step = max(1, chunk_size - overlap)

    while start < length:
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def _make_chunk_id(doc_type: str, section: str, index: int, chunk: str) -> str:
    base = f"{doc_type}:{section}:{index}:{chunk}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _iter_lines(text: str) -> Iterable[str]:
    for line in text.split("\n"):
        yield line
