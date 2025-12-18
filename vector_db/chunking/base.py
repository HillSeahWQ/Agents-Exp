"""
Base classes for document chunking.
Provides abstract interface for extensibility.
"""
import sys
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logging.logger import logging

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Types of chunks produced by chunkers."""
    TEXT = "text"
    TABLE = "table"
    IMAGE_HEAVY_PAGE = "image_heavy_page"
    MIXED = "mixed"
    CODE = "code"  # For future code chunking


@dataclass
class ChunkMetadata:
    """Base metadata class that all chunkers should use/extend."""
    source_file: str
    chunk_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        result = {"source_file": str(self.source_file), "chunk_id": self.chunk_id}
        
        # Add all other attributes
        for key, value in self.__dict__.items():
            if key not in result:
                if isinstance(value, Enum):
                    result[key] = value.value
                elif isinstance(value, Path):
                    result[key] = str(value)
                else:
                    result[key] = value
        
        return result


@dataclass
class Chunk:
    """Represents a processed document chunk."""
    content: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary format."""
        return {
            "content": self.content,
            "metadata": self.metadata.to_dict()
        }


class BaseChunker(ABC):
    """
    Abstract base class for document chunkers.
    All chunkers should inherit from this class.
    """
    
    @abstractmethod
    def chunk(self, file_path: str | Path) -> List[Chunk]:
        """
        Process a file and generate chunks.
        
        Args:
            file_path: Path to the file to chunk
            
        Returns:
            List of Chunk objects with content and metadata
        """
        pass
    
    @abstractmethod
    def get_metadata_schema(self) -> Dict[str, type]:
        """
        Return the metadata schema for this chunker.
        Used for automatic vector DB schema generation.
        
        Returns:
            Dictionary mapping field names to their types
        """
        pass
    
    def chunk_directory(self, directory: str | Path, extensions: List[str]) -> List[Chunk]:
        """
        Chunk all files in a directory with specified extensions.
        
        Args:
            directory: Path to directory
            extensions: List of file extensions to process (e.g., [".pdf", ".txt"])
            
        Returns:
            List of all chunks from all files
        """
        directory = Path(directory)
        all_chunks = []
        
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                chunks = self.chunk(path)
                all_chunks.extend(chunks)
        
        return all_chunks
    
    @classmethod
    def save_chunk_objects(cls, chunks: List[Chunk], output_path: Path | str):
        """Save Chunk objects directly to JSONL file."""
        contents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        cls.save_chunks_as_json(contents, metadatas, output_path)

    @classmethod
    def save_chunks_as_json(cls, contents, metadatas, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for c, m in zip(contents, metadatas):
            if hasattr(m, "to_dict"):
                m = m.to_dict()
            data.append({"content": c, "metadata": m})

        # write ALL in one go
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


    @classmethod
    def load_chunks(cls, input_path: Path | str) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Load chunks from JSON file.
        
        Args:
            input_path: Path to JSON file
            
        Returns:
            Tuple of (contents, metadatas)
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Chunk file not found: {input_path}")
        
        logger.info(f"Loading chunks from {input_path}...")
        
        with open(input_path, "r", encoding="utf-8") as f:
            chunk_dicts = json.load(f)
        
        contents = [chunk["content"] for chunk in chunk_dicts]
        metadatas = [chunk["metadata"] for chunk in chunk_dicts]
        
        logger.info(f"Loaded {len(contents)} chunks")
        
        return contents, metadatas

    @classmethod
    def get_chunk_statistics(cls, metadatas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics from chunk metadata.
        
        Args:
            metadatas: List of metadata dictionaries
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_chunks": len(metadatas),
            "chunk_types": {},
            "total_text_length": 0,
            "avg_text_length": 0,
            "total_tables": 0,
            "total_images": 0,
        }
        
        for metadata in metadatas:
            # Count chunk types
            chunk_type = metadata.get("chunk_type", "unknown")
            stats["chunk_types"][chunk_type] = stats["chunk_types"].get(chunk_type, 0) + 1
            
            # Sum text lengths
            text_length = metadata.get("text_length", 0)
            stats["total_text_length"] += text_length
            
            # Count tables and images
            stats["total_tables"] += metadata.get("num_tables", 0)
            stats["total_images"] += metadata.get("num_images", 0)
        
        # Calculate average
        if stats["total_chunks"] > 0:
            stats["avg_text_length"] = stats["total_text_length"] / stats["total_chunks"]
        
        return stats

    @classmethod
    def print_chunk_statistics(cls, metadatas: List[Dict[str, Any]]):
        """Print formatted chunk statistics."""
        stats = BaseChunker.get_chunk_statistics(metadatas)
        
        logger.info("="*80)
        logger.info("CHUNK STATISTICS")
        logger.info("="*80)
        logger.info(f"Total chunks: {stats['total_chunks']}")
        logger.info(f"Total text length: {stats['total_text_length']:,} characters")
        logger.info(f"Average text length: {stats['avg_text_length']:.0f} characters")
        logger.info("")
        logger.info("Chunk type distribution:")
        for chunk_type, count in stats["chunk_types"].items():
            percentage = (count / stats["total_chunks"]) * 100
            logger.info(f"  • {chunk_type}: {count} ({percentage:.1f}%)")
        logger.info("")
        logger.info(f"Total tables detected: {stats['total_tables']}")
        logger.info(f"Total images detected: {stats['total_images']}")
        logger.info("="*80)