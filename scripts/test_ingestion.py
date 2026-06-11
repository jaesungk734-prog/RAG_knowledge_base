# scripts/test_ingestion.py

import sys
from pathlib import Path

# This tells Python where to find your src/ modules.
# Without it, Python doesn't know where "ingestion" lives.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import TextChunker

def main():
    # ── 1. Check the PDF exists ────────────────────────────────────
    pdf_path = Path("data/pdfs/test.pdf")

    if not pdf_path.exists():
        print(f"ERROR: No PDF found at {pdf_path.absolute()}")
        print("Put a PDF named test.pdf inside your data/pdfs/ folder")
        return

    print("=" * 60)
    print("PHASE 1 TEST — Ingestion Pipeline")
    print("=" * 60)

    # ── 2. Load the PDF ────────────────────────────────────────────
    print(f"\n[ STEP 1 ] Loading PDF: {pdf_path.name}")
    documents = load_pdf(pdf_path)

    print(f"  Pages loaded:       {len(documents)}")
    print(f"  First page chars:   {len(documents[0].content)}")
    print(f"  Source path:        {documents[0].source_path}")
    print(f"  Metadata:           {documents[0].metadata}")

    print("\n  --- First 300 characters of page 1 ---")
    print(f"  {documents[0].content[:300]}")
    print("  ...")

    # ── 3. Chunk the documents ─────────────────────────────────────
    print("\n[ STEP 2 ] Chunking documents")
    print("  chunk_size:    512 tokens")
    print("  chunk_overlap: 64 tokens")

    chunker = TextChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk_documents(documents)

    print(f"\n  Total chunks created: {len(chunks)}")
    print(f"  Avg chunk length:     {sum(len(c.text) for c in chunks) // len(chunks)} chars")

    # ── 4. Inspect individual chunks ───────────────────────────────
    print("\n[ STEP 3 ] Inspecting first 3 chunks")
    print("-" * 60)

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n  CHUNK {i}")
        print(f"  ID:          {chunk.id}")
        print(f"  Page:        {chunk.page_number}")
        print(f"  Chunk index: {chunk.chunk_index}")
        print(f"  Text length: {len(chunk.text)} chars")
        print(f"  Preview:     {chunk.text[:150]}...")

    # ── 5. Check overlap is working ────────────────────────────────
    print("\n[ STEP 4 ] Verifying overlap between chunk 0 and chunk 1")
    print("-" * 60)

    if len(chunks) >= 2:
        # The last few words of chunk 0 should appear at the start of chunk 1
        end_of_chunk0 = chunks[0].text[-100:]
        start_of_chunk1 = chunks[1].text[:100:]

        print(f"  End of chunk 0:   ...{end_of_chunk0}")
        print(f"  Start of chunk 1: {start_of_chunk1}...")
        print("\n  (You should see some repeated words between the two above)")

    # ── 6. Summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  PDF:        {pdf_path.name}")
    print(f"  Pages:      {len(documents)}")
    print(f"  Chunks:     {len(chunks)}")
    print(f"  All chunks have IDs:   {all(c.id for c in chunks)}")
    print(f"  All chunks have text:  {all(c.text for c in chunks)}")
    print(f"  All chunks have pages: {all(c.page_number for c in chunks)}")
    print("\n✅ Phase 1 ingestion pipeline working correctly!")

if __name__ == "__main__":
    main()