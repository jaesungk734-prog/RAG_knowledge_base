# Personal RAG Knowledge Base

A retrieval-augmented generation system that lets you ask natural language questions about your own documents, PDFs and Markdown notes for now and get grounded answers with source citations.

Built as a learning project to understand how RAG pipelines actually work under the hood, without relying on frameworks like LangChain.

## How it works

1. Documents (PDF/Markdown) are loaded and split into overlapping chunks based on token count
2. Each chunk is embedded into a vector using OpenAI's `text-embedding-3-small`
3. Vectors are stored in Qdrant, a vector database, alongside metadata (source file, page number)
4. When you ask a question, it's embedded the same way and Qdrant returns the most semantically similar chunks
5. Those chunks are passed to GPT-4o as context, which generates an answer grounded in your documents and cites where the information came from

## Tech stack

- **FastAPI** — backend API, handles ingestion and query endpoints
- **Streamlit** — chat-based frontend
- **Qdrant** — vector database, run locally via Docker
- **OpenAI** — embeddings (`text-embedding-3-small`) and generation (`gpt-4o`)
- **pypdf / tiktoken** — PDF parsing and token-aware chunking

## Project structure
src/  
├── ingestion/      # PDF and Markdown loaders, token-based chunker  
├── embeddings/      # OpenAI embedding wrapper  
├── vectorstore/     # Qdrant client wrapper  
├── retrieval/       # Semantic search + context formatting  
├── generation/      # GPT-4o prompting and streaming  
└── api/             # FastAPI app, /ingest and /query routes  
frontend/  
└── app.py           # Streamlit chat interface  

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/jaesungk734-prog/RAG_knowledge_base.git
cd RAG_knowledge_base
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add your OpenAI API key**

Copy `.env.example` to `.env` and fill in your key:
OPENAI_API_KEY = sk-...

**3. Start Qdrant**

```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

**4. Run the backend**

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**5. Run the frontend** (in a separate terminal)

```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501`, upload a PDF or Markdown file, and start asking questions.

## Features

- PDF and Markdown ingestion
- Token-aware chunking with configurable overlap
- Semantic search via cosine similarity
- Streamed, grounded answers with source + page citations
- Source attribution shown in the UI for every answer

## Known limitations / planned improvements

- Re-ingesting the same file creates duplicate vectors (chunk IDs aren't deterministic yet)
- No re-ranking step — retrieval relies purely on embedding similarity
- Bookmark/URL ingestion not yet implemented
