
"""
Knowledge Agent.

Retrieves relevant fraud investigation guidance
from the persistent ChromaDB knowledge base.
"""

from src.agents.state.fraud_state import FraudState
from src.rag.retriever import retrieve_relevant_documents


def retrieve_policies(query: str, k: int = 3) -> list:
    """Retrieve relevant fraud policy chunks using semantic search."""

    return retrieve_relevant_documents(
        query=query,
        k=k,
    )


def knowledge_agent(state: FraudState) -> dict:
    """
    Retrieve relevant fraud policy evidence and update shared state.
    """

    # Step 1: Read information from shared state.
    prediction = state.get("prediction")
    investigation = state.get("investigation")

    if not prediction:
        raise ValueError("Prediction is missing from FraudState.")

    if not investigation:
        raise ValueError("Investigation report is missing from FraudState.")

    # Step 2: Prepare semantic search query.
    query = (
        f"Risk status: {prediction['risk_status']}\n"
        f"Fraud probability: {prediction['fraud_probability']}\n"
        f"Investigation findings:\n{investigation}"
    )

    # Step 3: Retrieve relevant policy evidence.
    print(
        "\n[DEBUG] Knowledge Agent: Starting policy retrieval...",
        flush=True,
    )

    print(
        "[DEBUG] Query length:",
        len(query),
        flush=True,
    )

    relevant_policies = retrieve_policies(
        query=query,
        k=3,
    )

    print(
        "[DEBUG] Knowledge Agent: Retrieval completed!",
        flush=True,
    )

    print(
        "[DEBUG] Retrieved documents:",
        len(relevant_policies),
        flush=True,
    )

    # Step 4: Format retrieved evidence.
    policy_sections = []

    for index, (document, distance) in enumerate(
        relevant_policies,
        start=1,
    ):
        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        chunk_index = document.metadata.get(
            "chunk_index",
            "Unknown",
        )

        policy_section = (
            f"Retrieved Evidence {index}\n"
            f"Source: {source}\n"
            f"Chunk Index: {chunk_index}\n"
            f"Retrieval Distance: {distance:.4f}\n"
            f"Content:\n{document.page_content}"
        )

        policy_sections.append(policy_section)

    # Step 5: Handle empty retrieval results.
    if policy_sections:
        policy_context = "\n\n".join(policy_sections)
    else:
        policy_context = (
            "No matching policy guidance was retrieved. "
            "Do not invent policy requirements."
        )

    # Step 6: Return updates to LangGraph shared state.
    return {
        "policy_context": policy_context,
        "status": "KNOWLEDGE_RETRIEVED",
    }
