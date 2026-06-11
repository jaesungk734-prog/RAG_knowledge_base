from qdrant_client import QdrantClient
from qdrant_client.models import(
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from loguru import logger

class QdrantStore:
    # we want to create a collection that stores chunks
    # and then search through vector

    def __init__(
            self,
            host: str = "localhost",
            port: int = 6333,
            collection_name: str = "knowledge-base",
            vector_size: int = 1536,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(host = host,port=port)
        logger.info(f"Connected to Qdrant at {host}:{port}")
    

    def create_collection(self,overwrite: bool = False):
        # check if collection exists, if not create

        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            if overwrite:
                logger.warning(f"Overwriting exisitng collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                logger.info(f"Collection '{self.collection_name}' already exists - skipping creation")
                return
        
        self.client.create_collection(
            collection_name = self.collection_name,
            vectors_config = VectorParams(
                size = self.vector_size,
                distance = Distance.COSINE,
            ),
        )
        logger.info(f"Created collection: {self.collection_name}")

    def store_chunks(self,chunks,embeddings: list[list[float]]):
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        points = [
            PointStruct(
                id = chunk.id,
                vector = embedding,
                payload = {
                    "text": chunk.text,
                    "source_path": chunk.source_path,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "doc_type": chunk.doc_type,
                    **chunk.metadata
                }
            )
            for chunk,embedding in zip(chunks,embeddings)
        ]

        self.client.upsert(
            collection_name = self.collection_name,
            points = points
        )

        logger.info(f"Stored {len(points)} points in '{self.collection_name}'")

    def search(self, query_vector:list[float],top_k: int = 5):
        # find the top_k most similar chunks to the query vector
        results = self.client.query_points(
            collection_name=self.collection_name,
            query = query_vector,
            limit = top_k,
            with_payload = True,
        )

        hits = []
        for point in results.points:
            payload = point.payload
            payload["score"] = round(point.score,4)
            hits.append(payload)
        return hits
    def count(self) -> int:
        # return the total number of stored vectors
        return self.client.count(collection_name = self.collection_name).count