import uuid
from dataclasses import dataclass, field
import tiktoken
from loguru import logger
from .pdf_loader import Document

@dataclass
class Chunk:
    # Chunk is what gets stored in Qdrant

    id: str
    text: str
    source_path: str
    page_number: int
    chunk_index: int
    doc_type: str
    metadata: dict = field(default_factory=dict)

class TextChunker:
    # this splits document objects into overlapping chunks

    def __init__(
            self,
            chunk_size: int = 512, # # of tokens
            chunk_overlap: int = 64, # shared tokens
            model: str = "text-embedding-3-small"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        #tiktoken not working yet but code is as follows

        try: 
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    
    def _split_into_chunks(self,text:str) -> list[str]:
        # the core logic
        # tokenizes the text into list of token ids
        # slide a window of chunk_size across the list
        # each step advances by the given overlap
        # decodes each window back to a string 
        token_ids = self.tokenizer.encode(text)

        if len(token_ids) == 0:
            return []

        chunks = []
        step = self.chunk_size - self.chunk_overlap

        start = 0
        # while loop so we don't miss the last window
        while start < len(token_ids):
            end = start + self.chunk_size
            window = token_ids[start:end]

            chunk_text = self.tokenizer.decode(window)
            chunks.append(chunk_text.strip())

            if end >= len(token_ids):
                break
            start += step
        
        return [c for c in chunks if c]
    
    def chunk_documents(self,documents: list[Document]) -> list[Chunk]:
        # turns a list of documents into a list of chunks
        all_chunks = []

        for doc in documents:
            raw_chunks = self._split_into_chunks(doc.content)

            for i, chunk_text in enumerate(raw_chunks):
                chunk = Chunk(
                    id = str(uuid.uuid4()),
                    text = chunk_text,
                    source_path = doc.source_path,
                    page_number = doc.page_number,
                    chunk_index = i,
                    doc_type = doc.doc_type,
                    metadata = {
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks_in_page": len(raw_chunks),
                    }
                )
                all_chunks.append(chunk)
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
        