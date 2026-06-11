# scripts/test_rag.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import TextChunker
from embeddings.embedder import Embedder
from vectorstore.qdrant_store import QdrantStore
from retrieval.retriever import Retriever
from generation.generator import Generator


def ingest_pdf(pdf_path: str, store: QdrantStore, embedder: Embedder):
    """
    Full ingestion pipeline — load, chunk, embed, store.
    We put this in a function so we can call it cleanly from main().
    """
    print(f"\n[ INGESTING ] {pdf_path}")

    docs = load_pdf(pdf_path)
    chunker = TextChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk_documents(docs)

    print(f"  Pages: {len(docs)} | Chunks: {len(chunks)}")
    print(f"  Embedding all {len(chunks)} chunks (this may take a few seconds)...")

    # Embed all chunks in one batch call — much faster than one by one
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_batch(texts)

    store.store_chunks(chunks, embeddings)
    print(f"  Stored {len(chunks)} vectors in Qdrant")


def ask(question: str, retriever: Retriever, generator: Generator):
    """
    Full RAG pipeline — retrieve context, generate answer, print it.
    """
    print(f"\n{'=' * 60}")
    print(f"QUESTION: {question}")
    print(f"{'=' * 60}")

    # Step 1: retrieve relevant chunks
    context, chunks = retriever.retrieve_and_format(question)

    if not chunks:
        print("\nNo relevant context found. Try a different question.")
        return

    # Show which sources were used
    print(f"\nSources retrieved ({len(chunks)} chunks):")
    for i, chunk in enumerate(chunks):
        import os
        filename = os.path.basename(chunk["source_path"])
        print(f"  {i+1}. {filename} | page {chunk['page_number']} | score {chunk['score']}")

    # Step 2: stream the answer
    print(f"\nANSWER:")
    print("-" * 60)

    # Stream tokens directly to the terminal as they arrive.
    # end="" prevents extra newlines, flush=True ensures
    # each token appears immediately rather than buffering.
    for token in generator.generate(question, context):
        print(token, end="", flush=True)

    print(f"\n{'-' * 60}")


def main():
    print("=" * 60)
    print("PHASE 3 TEST — Full RAG Pipeline")
    print("=" * 60)

    # ── Initialize components ──────────────────────────────────────
    embedder = Embedder()
    store = QdrantStore()
    store.create_collection(overwrite=True)   # fresh start for the test

    retriever = Retriever(
        embedder=embedder,
        store=store,
        top_k=5,
        min_score=0.3,
    )
    generator = Generator(model="gpt-4o")

    # ── Ingest your PDF ────────────────────────────────────────────
    ingest_pdf("data/pdfs/test.pdf", store, embedder)
    print(f"\n  Total vectors in DB: {store.count()}")

    # ── Ask questions ──────────────────────────────────────────────
    # These questions are answered using ONLY your PDF content.
    questions = [
        "What problem does the Transformer architecture solve?",
        "How does the attention mechanism work?",
        "What were the BLEU scores achieved by the model?",
    ]

    for question in questions:
        ask(question, retriever, generator)

    print(f"\n{'=' * 60}")
    print("✅ Phase 3 complete! Full RAG pipeline working.")
    print("=" * 60)


if __name__ == "__main__":
    main()