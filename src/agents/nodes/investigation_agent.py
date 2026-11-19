
"""
Investigation Agent.

Uses OpenAI through LangChain to investigate a transaction
based on existing XGBoost predictions and SHAP explanations.

This agent runs as a LangGraph node.
"""

import json
from functools import lru_cache

from src.agents.state.fraud_state import FraudState

from src.agents.fraud_investigation import (
    create_llm,
    create_investigation_prompt,
)


# ---------------------------------------------------------
# 1. INITIALIZE AND CACHE THE LLM
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def get_investigation_llm():
    """
    Initialize and cache the LangChain OpenAI model.

    The model is initialized once and reused for
    subsequent transaction investigations.
    """
    return create_llm()


# ---------------------------------------------------------
# 2. INVESTIGATION AGENT
# ---------------------------------------------------------

def investigation_agent(state: FraudState) -> dict:
    """
    LangGraph node responsible for AI investigation.

    Steps:
        1. Read transaction, prediction and SHAP evidence.
        2. Prepare structured evidence for the LLM.
        3. Build the existing investigation prompt.
        4. Invoke the OpenAI model.
        5. Return the investigation report to shared state.

    Args:
        state (FraudState): Current LangGraph state.

    Returns:
        dict: Investigation report and updated status.
    """

    # Step 1: Read evidence from shared state.
    transaction = state.get("transaction")
    prediction = state.get("prediction")
    shap_explanation = state.get("shap_explanation")

    # Validate required state fields.
    if not transaction:
        raise ValueError("Transaction data is missing.")

    if not prediction:
        raise ValueError("ML prediction is missing.")

    if not shap_explanation:
        raise ValueError("SHAP explanation is missing.")

    # Step 2: Prepare transaction evidence.
    transaction_evidence = {
    key: round(float(value), 6)
    for key, value in transaction.items()
}

    transaction_evidence["time_feature_note"] = (
        "Time represents elapsed seconds since the first "
        "transaction in the dataset, not clock time."
    )

    # Step 3: Prepare prediction evidence.
    prediction_evidence = {
        "fraud_probability_percentage": round(
            prediction["fraud_probability"] * 100,
            2,
        ),
        "fraud_probability": prediction["fraud_probability"],
        "decision_threshold": prediction["threshold"],
        "prediction": prediction["prediction"],
        "risk_status": prediction["risk_status"],
    }

    # Step 4: Prepare SHAP evidence.
    investigation_evidence = {
        "base_value": shap_explanation["base_value"],
        "top_features": shap_explanation["top_features"],
        "explanation_method": shap_explanation["explanation_method"],
        "output_note": shap_explanation["output_note"],
    }

    # Step 5: Get cached OpenAI model.
    llm = get_investigation_llm()

    # Step 6: Build the existing Phase 3 prompt.
    prompt = create_investigation_prompt()

    # Provide SHAP evidence to the prompt.
    prompt = prompt.partial(
        shap_data=json.dumps(
            investigation_evidence,
            indent=2,
        )
    )

    # Create LangChain processing chain.
    chain = prompt | llm

    # Step 7: Generate investigation report.
    response = chain.invoke(
        {
            "transaction_data": json.dumps(
                transaction_evidence,
                indent=2,
            ),
            "prediction_data": json.dumps(
                prediction_evidence,
                indent=2,
            ),
        }
    )

    # Step 8: Return updates to LangGraph state.
    return {
        "investigation": response.content,
        "status": "INVESTIGATION_COMPLETED",
    }
