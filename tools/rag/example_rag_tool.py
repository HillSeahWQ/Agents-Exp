"""
Generic FAISS RAG Search Tool
--------------------------------
Create new tools simply by copying this file and editing:

1. FAISS_INDEX_DIR
2. FAISS_INDEX_NAME

Everything else stays the same.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from langchain_core.tools import tool
from dotenv import load_dotenv

# Project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Local imports
from vector_db.vector_db.faiss_client import FAISSClient
from vector_db.embedding.embedding_manager import EmbeddingManager
from utils.logging.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ============================================================================
# CONFIG — ONLY EDIT THESE TWO WHEN REUSING
# ============================================================================

VECTOR_DB_DIR = project_root / "vector_db"

# CHANGE THESE 2 LINES WHEN CREATING A NEW TOOL FILE
FAISS_INDEX_DIR = VECTOR_DB_DIR / "data" / "faiss_indices"
FAISS_INDEX_NAME = "YOUR_FAISS_INDEX_NAME"  #TODO: TO_CHANGE, SAME AS IN Agent-H\vector_db\scripts\run_ingestion_faiss.py
# ALSO THE DOCSTRING OF THE SEARCH TOOL BELOW

# ============================================================================
# EMBEDDING CONFIG — KEEP CONSTANT ACROSS ALL RAG TOOLS
# ============================================================================
EMBEDDING_CONFIG = {
    "provider": "openai",
    "embedding_type": "text",
    "config": {
        "model": "text-embedding-3-large",
        "batch_size": 64,
        "normalize": True,
        "dimensions": 3072
    }
}

# Search config
DEFAULT_TOP_K = 10
NORMALIZE_QUERIES = True


# ============================================================================
# GENERIC SEARCH CLASS (Singleton)
# ============================================================================

class GenericFAISSRAGSearch:
    """Reusable FAISS RAG search engine."""
    
    _instance = None
    _faiss_client = None
    _embedder = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _initialize(self):
        """Lazy load FAISS + embedder once."""
        if self._faiss_client is None:
            logger.info(f"Initializing FAISS client for index: {FAISS_INDEX_NAME}")
            self._faiss_client = FAISSClient(
                index_dir=str(FAISS_INDEX_DIR),
                index_name=FAISS_INDEX_NAME,
            )

            # Load index
            if not self._faiss_client.load_index():
                msg = (
                    f"FAISS index '{FAISS_INDEX_NAME}' not found.\nExpected files in {FAISS_INDEX_DIR}:\n"
                    f"- {FAISS_INDEX_NAME}.index\n"
                    f"- {FAISS_INDEX_NAME}_metadata.pkl\n"
                    f"- {FAISS_INDEX_NAME}_contents.pkl"
                )
                logger.error(msg)
                raise FileNotFoundError(msg)

            logger.info("FAISS index loaded successfully.")

        if self._embedder is None:
            logger.info(
                f"Initializing embedder {EMBEDDING_CONFIG['provider']} - {EMBEDDING_CONFIG['config']['model']}"
            )
            self._embedder = EmbeddingManager.create_embedder(
                provider=EMBEDDING_CONFIG["provider"],
                embedding_type=EMBEDDING_CONFIG["embedding_type"],
                config=EMBEDDING_CONFIG["config"],
            )
            logger.info("Embedder initialized.")

    def search(self, query: str, top_k: int = DEFAULT_TOP_K,
               output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        
        self._initialize()

        logger.info(f"Searching FAISS index '{FAISS_INDEX_NAME}' for: {query}")

        query_embedding = self._embedder.embed([query])

        # Search FAISS
        results = self._faiss_client.search(
            query_embeddings=query_embedding,
            top_k=top_k,
            normalize=NORMALIZE_QUERIES,
            output_fields=output_fields,
        )

        return results[0] if results else []


# ============================================================================
# LANGCHAIN TOOL WRAPPER — FORMATTED OUTPUT
# ============================================================================

@tool
def search_documents(query: str) -> str:
    """
    [TODO: INPUT DESCRIPTION STRING OF WHAT CONTEXT WOULD REQUIRE THE SPECIFIC DOCUMENTS TO BE SEARCHED]
    """
    top_k = DEFAULT_TOP_K
    searcher = GenericFAISSRAGSearch()

    try:
        results = searcher.search(
            query=query,
            top_k=top_k,
            output_fields=["source_file", "page_number", "chunk_type", "content"],
        )

        if not results:
            return f"No results found for: '{query}'"

        out = f"Found {len(results)} relevant chunks for: '{query}'\n\n"
        for i, r in enumerate(results, 1):
            out += (
                f"--- Result {i} (Score: {r.get('score', 0):.4f}) ---\n"
                f"Source: {r.get('source_file')}\n"
                f"Page: {r.get('page_number')}\n"
                f"Type: {r.get('chunk_type')}\n"
                f"Content:\n{r.get('content')}\n\n"
            )
        return out

    except Exception as e:
        logger.error(e, exc_info=True)
        return f"Error: {e}"


# ============================================================================
# LANGCHAIN TOOL WRAPPER — STRUCTURED OUTPUT
# ============================================================================

@tool
def search_documents_detailed(query: str) -> List[Dict[str, Any]]:
    """
    Generic FAISS RAG search tool (structured).
    Use this when the agent needs dict output.
    """
    try:
        searcher = GenericFAISSRAGSearch()
        return searcher.search(query=query, top_k=DEFAULT_TOP_K)

    except Exception as e:
        return [{"error": str(e)}]
