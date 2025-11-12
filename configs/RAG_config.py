# config.py
from pathlib import Path

# Base directory = Allion.ai folder (parent of configs)
BASE_DIR = Path(__file__).resolve().parent.parent  # Go up one more level

# Paths (absolute, so working directory doesn't matter)
PDF_FOLDER = BASE_DIR / "docs" / "pdf_source"          # Folder containing PDFs
OUTPUT_DIR = BASE_DIR / "output"                        # All outputs go here
MARKDOWN_DIR = OUTPUT_DIR / "markdowns" 
CHUNKS_DIR = OUTPUT_DIR / "chunks"
CHROMA_DB_DIR = OUTPUT_DIR / "chroma_db"

# Vector Store Configuration
VECTOR_STORE_TYPE = "faiss"  # Options: "faiss", "chroma"
FAISS_INDEX_TYPE = "flat"    # Options: "flat", "ivf", "hnsw"

# FAISS Storage Paths
FAISS_DB_DIR = CHROMA_DB_DIR  # Add this line for compatibility
FAISS_INDEX_PATH = str(CHROMA_DB_DIR / "faiss_index")
FAISS_METADATA_PATH = str(CHROMA_DB_DIR / "faiss_metadata_v3.pkl")
FAISS_JSON_PATH = str(CHROMA_DB_DIR / "faiss_metadata_v3.json")

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-3-small"             # OpenAI embedding model
EMBEDDING_DIMENSION = 1536                             # Dimensions for text-embedding-3-small
EMBEDDING_BATCH_SIZE = 100                             # Batch size for embedding generation

# LLM Configuration
LLM_MODEL = "gpt-4o-mini"

# Processing Configuration
MAX_TOKENS_PER_CHUNK = 800
CHUNK_OVERLAP = 100

