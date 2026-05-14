from __future__ import annotations


def load_pdf_text(file_bytes: bytes) -> str:
    raise NotImplementedError("PDF parsing will be implemented in Phase 2.")


def load_docx_text(file_bytes: bytes) -> str:
    raise NotImplementedError("DOCX parsing will be implemented in Phase 2.")


def normalize_text(raw_text: str) -> str:
    raise NotImplementedError("Text normalization will be implemented in Phase 2.")
