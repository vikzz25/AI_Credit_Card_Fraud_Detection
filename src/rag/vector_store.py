
"""
Creates and manages the persistent ChromaDB vector store.

Stores fraud knowledge chunks, embeddings, and metadata
for semantic retrieval.
"""

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.rag.embeddings import get_embedding_model


# Resolve project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Persistent ChromaDB storage directory.
DEFAULT_VECTOR_STORE = PROJECT_ROOT / "vector_store"

# ChromaDB collection name.
COLLECTION_NAME = "fraud_knowledge"


def _generate_chunk_id(document: Document) -> str:
    """
    Generate a deterministic ID for a document chunk.

    This prevents duplicate records when the same chunks
    are added repeatedly.
    """

    unique_content = (
        f"{document.metadata.get('source', '')}"
        f"{document.metadata.get('chunk_index', '')}"
        f"{document.page_content}"
    )

    return hashlib.sha256(
        unique_content.encode("utf-8")
    ).hexdigest()


def create_vector_store(
    chunks: list[Document],
    persist_directory: str | Path = DEFAULT_VECTOR_STORE,
) -> Chroma:
    """
    Create or update a persistent ChromaDB vector store.

    Args:
        chunks: List of chunked LangChain Documents.
        persist_directory: Local database storage directory.

    Returns:
        A configured Chroma vector store instance.
    """

    if not chunks:
        raise ValueError("Cannot create vector store from empty chunks.")

    storage_path = Path(persist_directory)
    storage_path.mkdir(parents=True, exist_ok=True)

    # Initialize embedding model.
    embedding_model = get_embedding_model()

    # Connect to persistent ChromaDB collection.
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(storage_path),
    )

    # Generate stable IDs for every chunk.
    chunk_ids = [
        _generate_chunk_id(chunk)
        for chunk in chunks
    ]

    # Add or update documents in ChromaDB.
    vector_store.add_documents(
        documents=chunks,
        ids=chunk_ids,
    )

    return vector_store


def get_vector_store(
    persist_directory: str | Path = DEFAULT_VECTOR_STORE,
) -> Chroma:
    """
    Load an existing persistent ChromaDB vector store.

    Args:
        persist_directory: Directory containing ChromaDB data.

    Returns:
        An initialized Chroma vector store.

    Raises:
        FileNotFoundError: If the database directory does not exist.
        ValueError: If the collection contains no records.
    """

    storage_path = Path(persist_directory)

    if not storage_path.exists():
        raise FileNotFoundError(
            f"Vector store not found: {storage_path}"
        )

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(storage_path),
    )

    # Check whether the collection contains stored records.
    records = vector_store.get(include=[])

    if not records["ids"]:
        raise ValueError(
            "The fraud knowledge collection is empty."
        )

    return vector_store
