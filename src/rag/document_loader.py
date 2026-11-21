
"""
Loads fraud knowledge documents from the knowledge base.

Converts Markdown files into LangChain Document objects
for downstream RAG processing.
"""

from pathlib import Path

from langchain_core.documents import Document


# Resolve the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default knowledge base location.
DEFAULT_KNOWLEDGE_BASE = PROJECT_ROOT / "data" / "knowledge_base"


def load_documents(
    directory: str | Path = DEFAULT_KNOWLEDGE_BASE,
) -> list[Document]:
    """
    Load Markdown documents from the specified directory.

    Args:
        directory: Directory containing knowledge base documents.

    Returns:
        A list of LangChain Document objects.

    Raises:
        FileNotFoundError: If the knowledge base directory does not exist.
    """

    knowledge_path = Path(directory)

    if not knowledge_path.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {knowledge_path}"
        )

    if not knowledge_path.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, received: {knowledge_path}"
        )

    documents = []

    # Find all Markdown files recursively.
    markdown_files = sorted(knowledge_path.rglob("*.md"))

    for file_path in markdown_files:

        # Read document content.
        content = file_path.read_text(encoding="utf-8").strip()

        # Skip empty documents.
        if not content:
            continue

        # Convert content into a LangChain Document.
        document = Document(
            page_content=content,
            metadata={
                "source": str(file_path.relative_to(PROJECT_ROOT)),
                "file_name": file_path.name,
                "file_type": "markdown",
            },
        )

        documents.append(document)

    return documents
