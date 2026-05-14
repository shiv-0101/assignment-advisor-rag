import os

import streamlit as st


def get_config() -> dict:
    return {
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        "hf_api_key": os.getenv("HF_API_KEY", ""),
        "pinecone_api_key": os.getenv("PINECONE_API_KEY", ""),
        "pinecone_index": os.getenv("PINECONE_INDEX", ""),
        "embedding_model": os.getenv("HF_EMBEDDING_MODEL", ""),
        "embedding_dimension": os.getenv("EMBEDDING_DIMENSION", ""),
    }


def main() -> None:
    st.set_page_config(page_title="Assignment Advisor RAG", page_icon="📄")
    st.title("Assignment Advisor (RAG)")
    st.caption("Upload guidelines + draft, then ask a question about rubric compliance.")

    config = get_config()

    with st.sidebar:
        st.subheader("Settings")
        st.text_input("Pinecone index", value=config["pinecone_index"], disabled=True)
        st.text_input("Embedding model", value=config["embedding_model"], disabled=True)
        st.text_input("Embedding dimension", value=str(config["embedding_dimension"]), disabled=True)

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
        if not guidelines_file or not draft_file:
            st.error("Please upload both the Guidelines PDF and the Draft file.")
            return
        if not question.strip():
            st.error("Please enter a question.")
            return

        st.info("Phase 1 skeleton only. Processing pipeline will be wired next.")
        st.write("Question:", question)


if __name__ == "__main__":
    main()
