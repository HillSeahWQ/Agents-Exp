"""
Embedding module for generating vector embeddings.
"""
from .embedding.embedding_manager import EmbeddingManager
from .vector_db.faiss_client  import FAISSClient

__all__ = [
    "EmbeddingManager",
    "FAISSClient"
]