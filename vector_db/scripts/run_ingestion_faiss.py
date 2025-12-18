"""
FAISS ingestion script - loads chunks, generates embeddings, and ingests to FAISS.
Run this script after chunking to ingest documents into the FAISS vector database.
"""
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from vector_db.embedding.embedding_manager import EmbeddingManager
from vector_db.vector_db.faiss_client import FAISSClient
from utils.logging.logger import get_logger
from vector_db.chunking.base import BaseChunker

logger = get_logger(__name__)

# ======================================================================
# CONFIG (ALL CAPS)
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # AGENT-H/vector_db
DATA_DIR = PROJECT_ROOT / "data" # create a 'data' folder at AGENT-H/vector_db/data
INPUT_FOLDER_NAME = "test_pdfs" # TODO: EDIT THIS, INPUT FOLDER NAME - FOLDER WITH ALL THE DOCUMENTS
INPUT_DIR = DATA_DIR / INPUT_FOLDER_NAME # AGENT-H/vector_db/data/INPUT_FOLDER_NAME
CHUNKS_DIR = DATA_DIR / "chunks" # AGENT-H/vector_db/chunks
CHUNKS_OUTPUT_FILE_NAME = f"{INPUT_FOLDER_NAME}_chunks.json"
CHUNKS_OUTPUT_FILE_DIR = CHUNKS_DIR / CHUNKS_OUTPUT_FILE_NAME # AGENT-H/vector_db/chunks/CHUNKS_OUTPUT_FILE_NAME

ACTIVE_EMBEDDING_PROVIDER = "openai"  # or "sentence_transformers"
ACTIVE_EMBEDDING_TYPE = "text"
EMBEDDING_CONFIG = {
    "text": {
        # OpenAI embeddings
        "openai": {
            "model": "text-embedding-3-large",  # or "text-embedding-3-small"
            "batch_size": 64,
            "normalize": True,
            "dimensions": 3072  # 3072 for large, 1536 for small
        },
        # Sentence Transformers
        "sentence_transformers": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 64,
            "normalize": True,
            "dimensions": 384
        }
    }
}
FAISS_INDEX_DIR = DATA_DIR / "faiss_indices"
FAISS_INDEX_FILE_NAME = f"{INPUT_FOLDER_NAME}"
FAISS_CONFIG = {
    "index": {
        "index_dir": str(DATA_DIR / FAISS_INDEX_DIR),
        "name": FAISS_INDEX_FILE_NAME,
        "index_type": "Flat",  # Options: Flat, IVF, HNSW
        "metric_type": "IP",  # Options: IP (inner product), L2
        "normalize": True,  # True for cosine similarity with IP metric
        "params": {
            # For IVF: {"nlist": 100}
            # For HNSW: {"M": 32}
        }
    },
    "search": {
        "top_k": 5,
        "params": {
            # For IVF: {"nprobe": 10}
        }
    }
}
MILVUS_CONFIG = {
    "connection": {
        "host": "localhost",
        "port": "19530",
        "alias": "default"
    },
    "collection": {
        "name": "X_document_embeddings",
        "description": "Document embeddings with full chunk metadata"
    },
    "index": {
        "index_type": "IVF_FLAT",  # Options: HNSW, IVF_FLAT, IVF_PQ, etc.
        "metric_type": "IP",  # Options: IP (inner product), L2, COSINE
        "params": {
            "nlist": 1024  # For IVF_FLAT
            # For HNSW: {"M": 16, "efConstruction": 200}
        }
    },
    "search": {
        "top_k": 5,
        "params": {}  # Index-specific search params, e.g., {"nprobe": 10} for IVF
    }
}

def main():
    """Run FAISS ingestion pipeline WITHOUT argparse."""
    load_dotenv()

    # === FIXED PATHS (no CLI overriding) ===
    chunks_output_file_dir= CHUNKS_OUTPUT_FILE_DIR
    index_name = FAISS_CONFIG["index"]["name"]

    # === LOAD CONFIGS ===
    embedding_provider = ACTIVE_EMBEDDING_PROVIDER
    embed_config = EMBEDDING_CONFIG
    faiss_config = FAISS_CONFIG["index"].copy()

    # === LOG CONFIGURATION ===
    logger.info("=" * 80)
    logger.info("FAISS INGESTION PIPELINE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Chunks file: {chunks_output_file_dir}")
    logger.info(f"  Index name: {index_name}")
    logger.info(f"  Embedding provider: {embedding_provider}")
    logger.info(f"  Embedding model: {embed_config.get('model')}")
    logger.info(f"  Index type: {faiss_config['index_type']}")
    logger.info(f"  Metric type: {faiss_config['metric_type']}")
    logger.info(f"  Normalize: {faiss_config.get('normalize', False)}")
    logger.info("")

    # === VALIDATION ===
    if not chunks_output_file_dir.exists():
        logger.error(f"Chunks file not found: {chunks_output_file_dir}")
        logger.info("Run chunking first: python scripts/run_chunking.py")
        return 1

    try:
        # === LOAD CHUNKS ===
        logger.info("STEP 1: LOADING CHUNKS")
        logger.info("-" * 80)
        contents, metadatas = BaseChunker.load_chunks(chunks_output_file_dir)
        logger.info(f"Loaded {len(contents)} chunks")
        BaseChunker.print_chunk_statistics(metadatas)

        # === GENERATE EMBEDDINGS ===
        logger.info("")
        logger.info("STEP 2: GENERATING EMBEDDINGS")
        logger.info("-" * 80)

        embedder = EmbeddingManager.create_embedder(
            provider=embedding_provider,
            embedding_type=ACTIVE_EMBEDDING_TYPE,
            config=embed_config
        )

        logger.info(f"Model: {embedder.model_name}")
        logger.info(f"Dimension: {embedder.get_dimension()}")

        embeddings = embedder.embed(contents)
        logger.info(f"Generated embeddings: {embeddings.shape}")

        # === INITIALIZE FAISS ===
        logger.info("")
        logger.info("STEP 3: INITIALIZING FAISS")
        logger.info("-" * 80)

        client = FAISSClient(
            index_dir=faiss_config["index_dir"],
            index_name=index_name
        )

        # === CREATE INDEX ===
        logger.info("")
        logger.info("STEP 4: CREATING INDEX")
        logger.info("-" * 80)

        client.create_index(
            embedding_dim=embedder.get_dimension(),
            index_type=faiss_config["index_type"],
            metric_type=faiss_config["metric_type"],
            index_params=faiss_config.get("params", {}),
            drop_existing=False  # no CLI toggle
        )

        # === INGEST DATA ===
        logger.info("")
        logger.info("STEP 5: INGESTING DATA")
        logger.info("-" * 80)

        client.ingest_data(
            embeddings=embeddings,
            contents=contents,
            metadatas=metadatas,
            normalize=faiss_config.get("normalize", False)
        )

        # === VERIFY ===
        logger.info("")
        logger.info("STEP 6: VERIFYING INGESTION")
        logger.info("-" * 80)

        stats = client.get_index_stats()
        logger.info(f"Total vectors: {stats['num_vectors']:,}")
        logger.info(f"Index type: {stats['index_type']}")

        # === SUCCESS ===
        logger.info("")
        logger.info("=" * 80)
        logger.info("[SUCCESS] - INGESTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Index: {index_name}")
        logger.info(f"Vectors: {stats['num_vectors']:,}")
        logger.info("")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"[ERROR] - Ingestion failed: {e}")
        logger.exception("Full error traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())