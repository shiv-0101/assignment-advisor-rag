import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag.embeddings import embed_texts
from rag.generation import generate_feedback
from rag.ingest import (
    chunk_text,
    detect_headings,
    load_docx_text,
    load_pdf_text,
    normalize_text,
)
from rag.vectorstore import query_vectors, upsert_vectors


def get_config() -> dict:
    return {
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        "hf_api_key": os.getenv("HF_API_KEY", ""),
        "pinecone_api_key": os.getenv("PINECONE_API_KEY", ""),
        "pinecone_index": os.getenv("PINECONE_INDEX", ""),
        "embedding_model": os.getenv("HF_EMBEDDING_MODEL", ""),
        "embedding_dimension": os.getenv("EMBEDDING_DIMENSION", ""),
    }


def validate_config(config: dict) -> list[str]:
    missing = []
    if not config["hf_api_key"]:
        missing.append("HF_API_KEY")
    if not config["embedding_model"]:
        missing.append("HF_EMBEDDING_MODEL")
    if not config["pinecone_api_key"]:
        missing.append("PINECONE_API_KEY")
    if not config["pinecone_index"]:
        missing.append("PINECONE_INDEX")
    if not config["groq_api_key"]:
        missing.append("GROQ_API_KEY")
    return missing


def main() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    if not os.getenv("HF_API_KEY") or not os.getenv("HF_EMBEDDING_MODEL"):
        _load_env_fallback(env_path)
    st.set_page_config(page_title="Assignment Advisor RAG", page_icon="📄")
    st.title("Assignment Advisor (RAG)")
    st.caption("Upload guidelines + draft, then ask a question about rubric compliance.")

    config = get_config()

    with st.sidebar:
        st.subheader("Settings")
        st.text_input("Pinecone index", value=config["pinecone_index"], disabled=True)
        st.text_input("Embedding model", value=config["embedding_model"], disabled=True)
        st.text_input("Embedding dimension", value=str(config["embedding_dimension"]), disabled=True)
        chunk_size = st.number_input("Chunk size (chars)", min_value=800, max_value=4000, value=2200)
        overlap = st.number_input("Overlap (chars)", min_value=0, max_value=1000, value=250)
        top_k = st.number_input("Top-k per doc", min_value=1, max_value=10, value=4)
        missing_config = validate_config(config)
        if missing_config:
            st.warning(f"Missing config: {', '.join(missing_config)}")

    st.subheader("1) Upload documents")
    guidelines_file = st.file_uploader(
        "Guidelines PDF",
        type=["pdf"],
        accept_multiple_files=False,
    )
    draft_file = st.file_uploader(
        "Draft (PDF or DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=False,
    )

    st.subheader("2) Ask a question")
    question = st.text_input("Question", placeholder="Does my conclusion match the rubric?")
    submit = st.button("Check against guidelines")

    if submit:
        if missing_config:
            st.error("Missing required API keys or configuration. Check the sidebar.")
            return
        if not guidelines_file or not draft_file:
            st.error("Please upload both the Guidelines PDF and the Draft file.")
            return
        if not question.strip():
            st.error("Please enter a question.")
            return

        try:
            guidelines_bytes = guidelines_file.getvalue()
            draft_bytes = draft_file.getvalue()

            guidelines_raw = load_pdf_text(guidelines_bytes)
            draft_raw = (
                load_docx_text(draft_bytes)
                if draft_file.name.lower().endswith(".docx")
                else load_pdf_text(draft_bytes)
            )

            guidelines_text = normalize_text(guidelines_raw)
            draft_text = normalize_text(draft_raw)

            guidelines_headings = detect_headings(guidelines_text)
            draft_headings = detect_headings(draft_text)

            st.success("Text extracted and normalized.")
            st.write("Guidelines length:", len(guidelines_text))
            st.write("Draft length:", len(draft_text))

            with st.expander("Detected headings"):
                st.write("Guidelines:", guidelines_headings or "None detected")
                st.write("Draft:", draft_headings or "None detected")

            guidelines_chunks = chunk_text(
                guidelines_text,
                doc_type="guidelines",
                source_file=guidelines_file.name,
                chunk_size=int(chunk_size),
                overlap=int(overlap),
            )
            draft_chunks = chunk_text(
                draft_text,
                doc_type="draft",
                source_file=draft_file.name,
                chunk_size=int(chunk_size),
                overlap=int(overlap),
            )

            st.write("Guidelines chunks:", len(guidelines_chunks))
            st.write("Draft chunks:", len(draft_chunks))

            all_chunks = guidelines_chunks + draft_chunks
            texts = [chunk["text"] for chunk in all_chunks]

            with st.spinner("Embedding and uploading to Pinecone..."):
                vectors = embed_texts(texts)
                upsert_vectors(
                    [
                        (
                            chunk["chunk_id"],
                            vector,
                            {
                                "doc_type": chunk["doc_type"],
                                "section": chunk["section"],
                                "source_file": chunk["source_file"],
                                "text": chunk["text"],
                            },
                        )
                        for chunk, vector in zip(all_chunks, vectors)
                    ]
                )

            st.success("Chunks embedded and uploaded to Pinecone.")

            with st.spinner("Retrieving relevant context..."):
                query_vector = embed_texts([question])[0]
                draft_matches = query_vectors(
                    query_vector,
                    top_k=int(top_k),
                    filters={"doc_type": {"$eq": "draft"}, "section": {"$in": ["conclusion"]}},
                )
                guidelines_matches = query_vectors(
                    query_vector,
                    top_k=int(top_k),
                    filters={
                        "doc_type": {"$eq": "guidelines"},
                        "section": {"$in": ["requirements", "rubric", "grading"]},
                    },
                )

                if not draft_matches:
                    draft_matches = query_vectors(
                        query_vector,
                        top_k=int(top_k),
                        filters={"doc_type": {"$eq": "draft"}},
                    )
                if not guidelines_matches:
                    guidelines_matches = query_vectors(
                        query_vector,
                        top_k=int(top_k),
                        filters={"doc_type": {"$eq": "guidelines"}},
                    )

            draft_chunks_text = [
                match.get("metadata", {}).get("text", "") for match in draft_matches
            ]
            guidelines_chunks_text = [
                match.get("metadata", {}).get("text", "") for match in guidelines_matches
            ]

            with st.spinner("Generating feedback..."):
                feedback = generate_feedback(question, guidelines_chunks_text, draft_chunks_text)

            st.subheader("3) Feedback")
            st.write(feedback)
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
            return


if __name__ == "__main__":
    main()


def _load_env_fallback(env_path: Path) -> None:
    if not env_path.exists():
        return

    for encoding in ("utf-8-sig", "utf-16"):
        try:
            content = env_path.read_text(encoding=encoding)
            break
        except UnicodeError:
            continue
    else:
        return

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ[key.strip()] = value.strip()
