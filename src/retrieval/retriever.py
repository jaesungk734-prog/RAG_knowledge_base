import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))

from embeddings.embedder import Embedder
from vectorstore.qdrant_store import QdrantStore
from loguru import logger


class Retriever:
    # embed the question from user into a vector
    # serachs the Qdrant for similar chunks
    # format the chunks into a clean context string
    # for prompting later

    # separated so this only searches

    def __init__(
            self,
            embedder: Embedder,
            store: QdrantStore,
            top_k: int = 5,
            min_score: float = 0.3
    ):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.min_score = min_score
    
    def retrieve(self,question:str) -> list[dict]:
        # takes a english question and returns the top-k most relevant chunks

        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        logger.info(f"Retrieving context for : '{question}'")

        #embed question using same model
        query_vector = self.embedder.embed_text(question)

        #find closest vectors in Qdrant
        results = self.store.search(query_vector, top_k = self.top_k)

        #filter out results that are low conf
        filtered = [r for r in results if r["score"] >= self.min_score]

        if not filtered:
            logger.warning(
                f"No chunks above min_score = {self.min_score} "
                f"for question: '{question}'"
            )
        else:
            logger.info(f"Retrieved {len(filtered)} relevant chunks")
        
        return filtered

    def format_context(self,chunks: list[dict]) -> str:
        # format the received chunks into a single context string
        
        #format is [Source: pdf | Page: number | score]

        if not chunks:
            return "No relevant context found in your knowledge base"

        context = []
        for chunk in chunks:
            import os
            filename = os.path.basename(chunk.get("source_path","unknown"))

            block = (
                f"[SOURCE: {filename} | "
                f"PAGE: {chunk.get('page_number','?')} | "
                f"SCORE: {chunk.get('score',0)}]\n"
                f"{chunk['text']}"
            )
            context.append(block)

        return "\n\n---\n\n".join(context)
    

    def retrieve_and_format(self,question: str) ->tuple[str,list[dict]]:
        # combines function for convenience
        chunks = self.retrieve(question)
        context = self.format_context(chunks)
        return context,chunks
    