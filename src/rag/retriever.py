
"""
Retrieves relevant fraud knowledge using semantic search.

Uses ChromaDB to find policy chunks that are
semantically similar to a user's query.
"""

from langchain_core.documents import Document

from src.rag.vector_store import get_vector_store


def retrieve_relevant_documents(
    query: str,
    k: int = 3,
) -> list[tuple[Document, float]]:
    """
    Retrieve the most relevant knowledge chunks.

    Args:
        query: Natural language search query.
        k: Maximum number of relevant chunks to retrieve.

    Returns:
        A list of (Document, distance score) tuples.

        Lower distance generally indicates greater similarity.

    Raises:
        ValueError: If the query is empty or k is invalid.
    """

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    # Load the existing persistent vector store.
    vector_store = get_vector_store()

    # Perform semantic similarity search.
    results = vector_store.similarity_search_with_score(
        query=query,
        k=k,
    )

    return results
