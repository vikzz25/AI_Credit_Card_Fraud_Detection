"""
LangGraph workflow for AI Credit Card Fraud Detection.

Connects four specialized agents into one
sequential fraud investigation workflow.
"""

from langgraph.graph import StateGraph, START, END

from src.agents.state.fraud_state import FraudState

from src.agents.nodes.fraud_detection_agent import (
    fraud_detection_agent,
)

from src.agents.nodes.investigation_agent import (
    investigation_agent,
)

from src.agents.nodes.knowledge_agent import (
    knowledge_agent,
)

from src.agents.nodes.reporting_agent import (
    reporting_agent,
)


# ---------------------------------------------------------
# 1. BUILD THE WORKFLOW
# ---------------------------------------------------------

def build_fraud_investigation_graph():
    """
    Build and compile the fraud investigation workflow.

    Returns:
        CompiledStateGraph: Executable LangGraph workflow.
    """

    # Create a graph using our shared state schema.
    workflow = StateGraph(FraudState)

    # -----------------------------------------------------
    # REGISTER AGENTS AS GRAPH NODES
    # -----------------------------------------------------

    workflow.add_node(
        "fraud_detection",
        fraud_detection_agent,
    )

    workflow.add_node(
        "investigation",
        investigation_agent,
    )

    workflow.add_node(
        "knowledge",
        knowledge_agent,
    )

    workflow.add_node(
        "reporting",
        reporting_agent,
    )

    # -----------------------------------------------------
    # DEFINE EXECUTION ORDER
    # -----------------------------------------------------

    workflow.add_edge(
        START,
        "fraud_detection",
    )

    workflow.add_edge(
        "fraud_detection",
        "investigation",
    )

    workflow.add_edge(
        "investigation",
        "knowledge",
    )

    workflow.add_edge(
        "knowledge",
        "reporting",
    )

    workflow.add_edge(
        "reporting",
        END,
    )

    # -----------------------------------------------------
    # COMPILE GRAPH
    # -----------------------------------------------------

    compiled_graph = workflow.compile()

    return compiled_graph