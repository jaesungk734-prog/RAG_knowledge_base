import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parent.parent))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from embeddings.embedder import Embedder
from vectorstore.qdrant_store import QdrantStore
from ingestion.chunker import TextChunker
from retrieval.retriever import Retriever
from generation.generator import Generator
from api.routes.ingest import router as ingest_router
from api.routes.query import router as query_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    #initialize all expensive shared objects
    
    logger.info("Starting up Knowledge Base API...")

    app.state.embedder = Embedder()
    app.state.store = QdrantStore()
    app.state.chunker = TextChunker(chunk_size = 512, chunk_overlap=64)

    app.state.store.create_collection(overwrite = False)

    app.state.retriever = Retriever(
        embedder = app.state.embedder,
        store = app.state.store,
        top_k = 5,
        min_score = 0.3,
    )

    app.state.generator = Generator(model = "gpt-4o")

    logger.info("All components initialized. Server ready.")

    yield #server starts

    logger.info("Shutting down Knowledge Base API...")

app = FastAPI(
    title = "Personal Knowledge Base API",
    description = "RAG-powered API for querying personal documents",
    version = "0.1.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

app.include_router(ingest_router)
app.include_router(query_router)


@app.get("/health")
async def health_check():
    #check if server is running

    try: 
        count = app.state.store.count()
        return {
            "status" : "ok",
            "vectors_stored" : count,
            "model": "gpt-4o",
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e),
        }