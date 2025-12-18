"""
X Document RAG Tool
Searches FAISS vector database for relevant document chunks.

FIXED PARAMS, not dynamically chosen by LLM
1. Database - Which specific DB/Table to search
2. Top K - Fixed here in config, avoid LLM deciding
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from langchain_core.tools import tool
from dotenv import load_dotenv

# Add parent directories to path to import from vector_db
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from vector_db.vector_db.faiss_client import FAISSClient
from vector_db.embedding.embedding_manager import EmbeddingManager
from utils.logging.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths - adjust based on your repo structure
VECTOR_DB_DIR = project_root / "vector_db"
FAISS_INDEX_DIR = VECTOR_DB_DIR / "data" / "faiss_indices"
FAISS_INDEX_NAME = "company_x_docs"

# Embedding configuration - must match what was used during ingestion
EMBEDDING_CONFIG = {
    "provider": "openai",  # or "sentence_transformers"
    "embedding_type": "text",
    "config": {
        "model": "text-embedding-3-large",
        "batch_size": 64,
        "normalize": True,
        "dimensions": 3072
    }
}

# Search configuration
DEFAULT_TOP_K = 10
NORMALIZE_QUERIES = True  # Should match ingestion normalization


# ============================================================================
# RAG TOOL IMPLEMENTATION
# ============================================================================

class XRAGSearch:
    """
    Singleton class to manage FAISS client and embedder instances.
    Avoids reloading the index and embedder on every tool call.
    """
    _instance = None
    _faiss_client = None
    _embedder = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _initialize(self):
        """Lazy initialization of FAISS client and embedder."""
        if self._faiss_client is None:
            logger.info(f"Initializing FAISS client for index: {FAISS_INDEX_NAME}")
            # Initialize FAISS client
            self._faiss_client = FAISSClient(
                index_dir=str(FAISS_INDEX_DIR),
                index_name=FAISS_INDEX_NAME
            )
            
            # Load the index
            if not self._faiss_client.load_index():
                error_msg = (
                    f"FAISS index '{FAISS_INDEX_NAME}' not found at {FAISS_INDEX_DIR}. "
                    f"Please ensure the following files exist:\n"
                    f"  - {FAISS_INDEX_NAME}.index\n"
                    f"  - {FAISS_INDEX_NAME}_metadata.pkl\n"
                    f"  - {FAISS_INDEX_NAME}_contents.pkl"
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            logger.info(f"FAISS index loaded successfully: {self._faiss_client.get_index_stats()}")
        
        if self._embedder is None:
            logger.info(f"Initializing embedder: {EMBEDDING_CONFIG['provider']} - {EMBEDDING_CONFIG['config']['model']}")
            # Initialize embedder
            self._embedder = EmbeddingManager.create_embedder(
                provider=EMBEDDING_CONFIG["provider"],
                embedding_type=EMBEDDING_CONFIG["embedding_type"],
                config=EMBEDDING_CONFIG["config"]
            )
            logger.info("Embedder initialized successfully")
    
    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search X documents for relevant chunks.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            output_fields: Specific metadata fields to return (None = all)
            
        Returns:
            List of matching document chunks with metadata and scores
        """
        # Ensure initialization
        self._initialize()
        
        logger.info(f"Searching for query: '{query}' (top_k={top_k})")
        
        # Embed the query
        query_embedding = self._embedder.embed([query])  # Returns numpy array
        logger.debug(f"Query embedding shape: {query_embedding.shape}")
        
        # Search FAISS
        results = self._faiss_client.search(
            query_embeddings=query_embedding,
            top_k=top_k,
            normalize=NORMALIZE_QUERIES,
            output_fields=output_fields
        )
        
        # Log retrieved chunks
        retrieved_results = results[0] if results else []
        logger.info(f"Retrieved {len(retrieved_results)} chunks for query: '{query}'")
        
        for i, result in enumerate(retrieved_results, 1):
            source = result.get("source_file", "Unknown")
            page = result.get("page_number", "N/A")
            score = result.get("score", 0)
            chunk_id = result.get("id", "N/A")
            content_preview = result.get("content", "")[:100] + "..." if len(result.get("content", "")) > 100 else result.get("content", "")
            
            logger.info(
                f"  Chunk {i}: ID={chunk_id} | Score={score:.4f} | "
                f"Source={source} | Page={page} | Preview='{content_preview}'"
            )
        
        # Return first query's results (we only searched one query)
        return retrieved_results


# ============================================================================
# LANGCHAIN TOOL WRAPPER
# ============================================================================

@tool
def search_X_documents(
    query: str
) -> str:
    """
    Search Company X's documentation for relevant information.
    
    Use this tool when you need to find information about:
    - Company X's benefits, policies, or procedures
    - Healthcare coverage, surgeries, hospitals
    - Employee benefits and insurance
    - Any Company X's specific documentation
    
    Args:
        query: The search query describing what information you need
    
    Returns:
        A formatted string containing the most relevant document chunks with source information
    
    Example:
        search_X_documents("What does X cover for surgeries?", top_k=3)
    """
    # Validate top_k
    top_k = max(1, min(DEFAULT_TOP_K, 20))  # Clamp between 1 and 20
    
    logger.info(f"Tool called: search_X_documents(query='{query}', top_k={top_k})")
    
    try:
        # Perform search
        searcher = XRAGSearch()
        results = searcher.search(
            query=query,
            top_k=top_k,
            output_fields=["source_file", "page_number", "chunk_type", "content"]
        )
        
        if not results:
            logger.warning(f"No results found for query: '{query}'")
            return f"No relevant documents found for query: '{query}'"
        
        # Format results for LLM consumption
        formatted_output = f"Found {len(results)} relevant document chunks for: '{query}'\n\n"
        
        for i, result in enumerate(results, 1):
            score = result.get("score", 0)
            source = result.get("source_file", "Unknown")
            page = result.get("page_number", "N/A")
            chunk_type = result.get("chunk_type", "text")
            content = result.get("content", "")
            
            formatted_output += f"--- Result {i} (Relevance Score: {score:.4f}) ---\n"
            formatted_output += f"Source: {source} | Page: {page} | Type: {chunk_type}\n"
            formatted_output += f"Content:\n{content}\n\n"
        
        logger.info(f"Successfully formatted {len(results)} results for query: '{query}'")
        return formatted_output
    
    except FileNotFoundError as e:
        logger.error(f"FAISS index not found: {e}")
        return f"Error: FAISS index not found. {str(e)}"
    except Exception as e:
        logger.error(f"Error searching documents: {e}", exc_info=True)
        return f"Error searching documents: {str(e)}"


# ============================================================================
# ALTERNATIVE: More detailed tool with structured output
# ============================================================================

@tool
def search_X_documents_detailed(
    query: str
) -> List[Dict[str, Any]]:
    """
    Search X documentation and return structured results.
    
    This version returns structured data instead of formatted text,
    useful if the agent needs to process results programmatically.
    
    Args:
        query: The search query by the user
    
    Returns:
        List of dictionaries containing:
        - content: The document chunk text
        - score: Relevance score
        - source_file: Source document name
        - page_number: Page number in source
        - chunk_type: Type of chunk (text, table, etc.)
        - preview: Short preview of content
    """
    top_k = max(1, min(DEFAULT_TOP_K, 20))
    
    try:
        searcher = XRAGSearch()
        results = searcher.search(query=query, top_k=top_k)
        return results
    
    except Exception as e:
        return [{"error": str(e)}]