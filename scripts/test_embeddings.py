# scripts/test_embeddings.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import TextChunker
from embeddings.embedder import Embedder
from vectorstore.qdrant_store import QdrantStore


def main():
    print("=" * 60)
    print("PHASE 2 TEST — Embeddings + Qdrant")
    print("=" * 60)

    # ── 1. Ingest (same as Phase 1) ───────────────────────────────
    print("\n[ STEP 1 ] Loading and chunking PDF...")
    docs = load_pdf("data/pdfs/test.pdf")
    chunker = TextChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk_documents(docs)
    print(f"  Chunks ready: {len(chunks)}")

    # ── 2. Embed ───────────────────────────────────────────────────
    print("\n[ STEP 2 ] Embedding chunks with OpenAI...")
    embedder = Embedder()

    # Only embed first 5 chunks for the test — saves money and time.
    # A full PDF would cost ~$0.001 total anyway, but no need to wait.
    test_chunks = chunks[:5]
    texts = [chunk.text for chunk in test_chunks]
    embeddings = embedder.embed_batch(texts)

    print(f"  Chunks embedded:    {len(embeddings)}")
    print(f"  Vector dimensions:  {len(embeddings[0])}")
    print(f"  First 5 values:     {[round(v, 4) for v in embeddings[0][:5]]}")

    # ── 3. Store in Qdrant ─────────────────────────────────────────
    print("\n[ STEP 3 ] Storing in Qdrant...")
    store = QdrantStore()
    store.create_collection(overwrite=True)
    store.store_chunks(test_chunks, embeddings)
    print(f"  Total vectors in DB: {store.count()}")

    # ── 4. Search ──────────────────────────────────────────────────
    print("\n[ STEP 4 ] Running a test search...")
    query = "What is the attention mechanism?"
    print(f"  Query: '{query}'")

    query_vector = embedder.embed_text(query)
    results = store.search(query_vector, top_k=3)

    print(f"\n  Top {len(results)} results:\n")
    for i, result in enumerate(results):
        print(f"  RESULT {i + 1}")
        print(f"  Score:   {result['score']}  (higher = more relevant)")
        print(f"  Page:    {result['page_number']}")
        print(f"  Preview: {result['text'][:150]}...")
        print()

    print("=" * 60)
    print("✅ Phase 2 complete! Embeddings and Qdrant working.")
    print("=" * 60)


if __name__ == "__main__":
    main()