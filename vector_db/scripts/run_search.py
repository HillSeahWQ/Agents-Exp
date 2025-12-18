import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from vector_db.embedding.embedding_manager import EmbeddingManager
from vector_db.vector_db.milvus_client import MilvusClient
from vector_db.vector_db.faiss_client import FAISSClient
from utils.logging.logger import get_logger

logger = get_logger(__name__)

# ======================================================================
# CONFIG (ALL CAPS)
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # AGENT-H/vector_db
DATA_DIR = PROJECT_ROOT / "data" # create a 'data' folder at AGENT-H/vector_db/data
INPUT_FOLDER_NAME = "company_x_docs" # TODO: EDIT THIS, INPUT FOLDER NAME - FOLDER WITH ALL THE DOCUMENTS

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
FAISS_INDEX_FILE_NAME = f"{INPUT_FOLDER_NAME}_faiss"
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
        "top_k": 10,
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
        "top_k": 10,
        "params": {}  # Index-specific search params, e.g., {"nprobe": 10} for IVF
    }
}
ACTIVE_VECTOR_DB = "faiss"

EVAL_DIR = DATA_DIR / "evaluation"
QUERY_RESULTS_DIR = EVAL_DIR / "query_results"

QUERIES = [
    {"query_id": "q1", "query_text": "How much does X cover for surgeries"},
    {"query_id": "q2", "query_text": "What are the hospitals covered?"} 
]

SAVE_RESULTS = True       # True to save automatically
SAVE_RESULTS_PATH = QUERY_RESULTS_DIR / f"{INPUT_FOLDER_NAME}_query_results.json"

# ======================================================================
# QUERY EXECUTION FUNCTIONS
# ======================================================================
def save_results(results: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved successfully")


def load_query_list():
    """Return queries and IDs from the QUERIES config."""
    queries = [q["query_text"] for q in QUERIES]
    query_ids = [q["query_id"] for q in QUERIES]
    return queries, query_ids


def query_faiss(query_embeddings, output_fields):
    index_name = FAISS_CONFIG["index"]["name"]
    index_dir = FAISS_CONFIG["index"]["index_dir"]

    client = FAISSClient(index_dir=index_dir, index_name=index_name)
    logger.info(f"Loading FAISS index: {index_name}")

    if not client.load_index():
        raise FileNotFoundError(
            f"FAISS index '{index_name}' not found. Did you run ingestion?"
        )

    top_k = FAISS_CONFIG["search"]["top_k"]
    normalize = FAISS_CONFIG["index"].get("normalize", False)

    return client.search(
        query_embeddings=query_embeddings,
        top_k=top_k,
        normalize=normalize,
        search_params=FAISS_CONFIG["search"].get("params", {}),
        output_fields=output_fields
    )


def query_milvus(query_embeddings, output_fields):
    milvus_conn = MILVUS_CONFIG["connection"]
    collection = MILVUS_CONFIG["collection"]["name"]

    client = MilvusClient(
        host=milvus_conn["host"],
        port=milvus_conn["port"],
        alias=milvus_conn["alias"]
    )

    client.connect()
    top_k = MILVUS_CONFIG["search"]["top_k"]

    results = client.search(
        collection_name=collection,
        query_embeddings=query_embeddings,
        top_k=top_k,
        metric_type=MILVUS_CONFIG["index"]["metric_type"],
        search_params=MILVUS_CONFIG["search"]["params"],
        output_fields=output_fields
    )

    client.disconnect()
    return results


# ======================================================================
# MAIN LOGIC (NO ARGS)
# ======================================================================

def main():
    load_dotenv()

    logger.info("=" * 80)
    logger.info(f"RUNNING QUERY PIPELINE ({ACTIVE_VECTOR_DB.upper()})")
    logger.info("=" * 80)

    # -------------------------------
    # 1. Load queries from config
    # -------------------------------
    queries, query_ids = load_query_list()

    # -------------------------------
    # 2. Build embedder
    # -------------------------------
    embed_cfg = EMBEDDING_CONFIG[ACTIVE_EMBEDDING_TYPE][ACTIVE_EMBEDDING_PROVIDER]

    embedder = EmbeddingManager.create_embedder(
        provider=ACTIVE_EMBEDDING_PROVIDER,
        embedding_type=ACTIVE_EMBEDDING_TYPE,
        config=embed_cfg
    )

    logger.info("Embedding queries...")
    query_embeddings = embedder.embed(queries)

    # -------------------------------
    # 3. Query Vector DB
    # -------------------------------
    output_fields = [
        "source_file", "id", "page_number",
        "chunk_type", "preview", "content"
    ]

    if ACTIVE_VECTOR_DB == "faiss":
        results = query_faiss(query_embeddings, output_fields)
        collection_name = FAISS_CONFIG["index"]["name"]

    elif ACTIVE_VECTOR_DB == "milvus":
        results = query_milvus(query_embeddings, output_fields)
        collection_name = MILVUS_CONFIG["collection"]["name"]

    else:
        raise ValueError(f"Unknown vector DB: {ACTIVE_VECTOR_DB}")

    # -------------------------------
    # 4. Build structured output
    # -------------------------------
    structured_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "vector_db": ACTIVE_VECTOR_DB,
            "embedding_provider": ACTIVE_EMBEDDING_PROVIDER,
            "embedding_model": embedder.model_name,
            "collection_name": collection_name,
            "top_k": (
                FAISS_CONFIG["search"]["top_k"]
                if ACTIVE_VECTOR_DB == "faiss"
                else MILVUS_CONFIG["search"]["top_k"]
            ),
            "num_queries": len(queries)
        },
        "queries": []
    }

    for qid, qtext, qres in zip(query_ids, queries, results):
        structured_results["queries"].append({
            "query_id": qid,
            "query_text": qtext,
            "results": qres
        })

    # -------------------------------
    # 5. Save results optional
    # -------------------------------
    if SAVE_RESULTS:
        save_results(structured_results, SAVE_RESULTS_PATH)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suggested = QUERY_RESULTS_DIR / f"query_results_{ACTIVE_VECTOR_DB}_{ts}.json"
        logger.info(f"To save results, set SAVE_RESULTS=True or output: {suggested}")

    logger.info("[SUCCESS] Query complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())