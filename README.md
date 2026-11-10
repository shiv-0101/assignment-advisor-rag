# Assignment Advisor RAG

An AI-powered Retrieval-Augmented Generation (RAG) assistant designed to help students and educators evaluate academic writing drafts against specific assignment guidelines. The application compares draft content with guidelines using custom vector search and leverages Groq's fast LLM inference engine to provide structured, actionable feedback.

---

## 🌟 Key Features

- **Document Ingestion & Indexing:** Chunks and embeds assignment guidelines and student drafts into a vector index.
- **Contextual Retrieval:** Performs similarity searches to extract relevant excerpts from guidelines and draft documents.
- **Groq LLM Generation:** Uses Groq's OpenAI-compatible chat API (`llama3-8b-8192` by default) for fast inference.
- **Structured Academic Feedback:** Generates clear, bulleted feedback highlighting missing points, mismatches, and suggested improvements.

---

## 📁 Repository Structure

```text
assignment-advisor-rag/
│
├── rag/
│   ├── embeddings.py      # Embedding generation logic
│   ├── generation.py      # Groq API integration and prompt construction
│   ├── ingest.py          # Data ingestion and document chunking
│   └── vectorstore.py     # Vector database setup and similarity search
│
├── app.py                 # Main entry point (Streamlit / API interface)
├── requirements.txt       # Project dependencies
└── .gitignore             # Ignored files and directories

```
🚀 Getting Started
------------------

### Prerequisites

*   Python 3.8+
    
*   A valid **Groq API Key** ([Get one here](https://console.groq.com/))
