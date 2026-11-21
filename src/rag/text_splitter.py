
"""
Splits fraud knowledge documents into smaller chunks.

Uses LangChain's RecursiveCharacterTextSplitter
to prepare documents for embedding and vector storage.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Split documents into smaller overlapping chunks.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters.

    Returns:
        A list of chunked LangChain Document objects.

    Raises:
        ValueError: If chunking parameters are invalid.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    # Initialize the recursive text splitter.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    # Split documents while preserving their metadata.
    chunks = text_splitter.split_documents(documents)

    # Assign a unique sequential index to every chunk.
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    return chunks
