import os
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

# loads the .env file using openAI key
load_dotenv()

class Embedder:
    #converts text into vectors using embedding model

    def __init__(self,model: str = "text-embedding-3-small"):
        self.model = model

        api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found"
                "Make sure it's set in your .env file and load_dotenv() was called."
            )
        
        self.client = OpenAI(api_key = api_key)

        logger.info(f"Embedder initialized with model: {self.model}")

    def embed_text(self,text:str) -> list[float]:
        # converts a string into a vector
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        # API call
        response = self.client.embeddings.create(
            model = self.model,
            input = text,
        )

        return response.data[0].embedding
    
    def embed_batch(self,texts: list[str]) -> list[list[float]]:
        # converts multiple strings into vectors in a single API call

        if not texts:
            return []

        clean_texts = [t.strip() for t in texts if t and t.strip()]

        if not clean_texts:
            return []

        logger.info(f"Embedding batch of {len(clean_texts)} texts...")

        response = self.client.embeddings.create(
            model = self.model,
            input = clean_texts,
        )

        embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

        logger.info(f"Successfully embedded {len(embeddings)} texts")

        return embeddings

