"""
CoCo Ingestion Configuration
============================
Configuration settings for the standalone ingestion module.
"""

from pathlib import Path

# Project paths
MODULE_ROOT = Path(__file__).parent.parent  # INGESTION_MODULE/

# Default documents path — within INGESTION_MODULE itself
# Users should override via CLI argument: python ingest.py <path>
DEFAULT_DOCUMENTS_PATH = MODULE_ROOT / "documents_to_ingest"

# Ingestion settings
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# Required files for a valid vector store package
REQUIRED_VECTOR_STORE_FILES = [
    "index.faiss",
    "index.pkl"
]

# Optional files that may be included
OPTIONAL_VECTOR_STORE_FILES = [
    "document_registry.json"
]
