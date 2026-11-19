
"""
Knowledge Agent.

Retrieves relevant fraud investigation guidance
from the temporary policy knowledge base.

This implementation will be upgraded to RAG
with ChromaDB in Phase 5.
"""

from src.agents.state.fraud_state import FraudState

from src.rag.policy_knowledge import FRAUD_POLICIES


# ---------------------------------------------------------
# 1. RETRIEVE RELEVANT POLICIES
# ---------------------------------------------------------

def retrieve_policies(query: str) -> list:
    """
    Retrieve relevant policies using simple keyword matching.

    Args:
        query (str): Investigation context.

    Returns:
        list: Relevant policy documents.
    """

    if not query or not query.strip():
        return []

    query_words = set(query.lower().split())

    scored_policies = []

    for policy in FRAUD_POLICIES:

        searchable_text = (
            policy["title"] + " " + policy["category"]
            + " " + policy["content"]
        ).lower()

        policy_words = set(searchable_text.split())

        score = len(query_words.intersection(policy_words))

        if score > 0:
            scored_policies.append((score, policy))

    # Sort by relevance score.
    scored_policies.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        policy
        for score, policy in scored_policies[:3]
    ]


# ---------------------------------------------------------
# 2. KNOWLEDGE AGENT
# ---------------------------------------------------------

def knowledge_agent(state: FraudState) -> dict:
    """
    LangGraph node responsible for retrieving
    relevant investigation guidance.

    Args:
        state (FraudState): Current shared workflow state.

    Returns:
        dict: Retrieved policy context and updated status.
    """

    prediction = state.get("prediction")
    investigation = state.get("investigation")

    if not prediction:
        raise ValueError(
            "Prediction is missing from FraudState."
        )

    if not investigation:
        raise ValueError(
            "Investigation report is missing from FraudState."
        )

    # Prepare search context.
    query = (
        f"Risk status: {prediction['risk_status']} "
        f"Fraud probability: {prediction['fraud_probability']} "
        f"Investigation: {investigation}"
    )

    # Retrieve relevant policy documents.
    relevant_policies = retrieve_policies(query)

    # Format policy context.
    policy_context = "\n\n".join(
        (
            f"Policy ID: {policy['id']}\n"
            f"Title: {policy['title']}\n"
            f"Category: {policy['category']}\n"
            f"Guidance: {policy['content']}"
        )
        for policy in relevant_policies
    )

    if not policy_context:
        policy_context = (
            "No matching policy guidance was retrieved. "
            "Do not invent policy requirements."
        )

    return {
        "policy_context": policy_context,
        "status": "KNOWLEDGE_RETRIEVED",
    }
