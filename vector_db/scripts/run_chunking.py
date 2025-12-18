"""
Document chunking script with config-only (CAPS) variables, no CLI args.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from vector_db.chunking.pdf_chunker import MultimodalPDFChunker, BaseChunker
from utils.logging.logger import get_logger

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

CHUNKING_CONFIG = {
    "PDF": {
        "image_coverage_threshold": 0.15,
        "vision_model": "gpt-4o-mini",
        "log_level": "INFO",
    }
}

# ======================================================================
# MAIN
# ======================================================================

def main():
    """Run chunking pipeline using config variables only."""

    input_dir = INPUT_DIR
    chunks_output_file_dir = CHUNKS_OUTPUT_FILE_DIR
    chunking_config = CHUNKING_CONFIG["PDF"].copy()

    # Create output directory
    chunks_output_file_dir.parent.mkdir(parents=True, exist_ok=True)

    # === LOG CONFIGURATION ===
    logger.info("=" * 80)
    logger.info("DOCUMENT CHUNKING PIPELINE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Input directory: {input_dir}")
    logger.info(f"  Output file: {chunks_output_file_dir}")
    logger.info(f"  Image threshold: {chunking_config['image_coverage_threshold']}")
    logger.info(f"  Vision model: {chunking_config['vision_model']}")
    logger.info("")

    # === VALIDATION ===
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    # === RUN CHUNKING ===
    try:
        logger.info("Initializing PDF chunker...")
        chunker = MultimodalPDFChunker(**chunking_config)

        pdf_files = list(input_dir.glob("**/*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return 1

        logger.info(f"Found {len(pdf_files)} PDF files")
        logger.info("")

        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file.name}")
            try:
                all_chunks = chunker.chunk(pdf_file)
                logger.info(f"Generated {len(all_chunks)} chunks + metadata")
            except Exception as e:
                logger.error(f"Failed: {e}")
                continue

        # Save chunks
        logger.info("")
        BaseChunker.save_chunk_objects(all_chunks, chunks_output_file_dir)

        logger.info("")
        logger.info("=" * 80)
        logger.info("[SUCCESS] - CHUNKING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total chunks: {len(all_chunks)}")
        logger.info(f"Output saved to: {chunks_output_file_dir}")
        logger.info("")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"[ERROR] - Chunking failed: {e}")
        logger.exception("Full error traceback:")
        return 1


if __name__ == "__main__":
    load_dotenv()
    sys.exit(main())