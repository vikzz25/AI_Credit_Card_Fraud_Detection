
"""
Fraud Detection Agent.

Uses the trained XGBoost model to predict fraud risk
and SHAP to explain the prediction.

This agent is designed to run as a LangGraph node.
"""

from functools import lru_cache

from src.agents.state.fraud_state import FraudState

from src.agents.fraud_investigation import (
    load_fraud_model,
    predict_transaction,
)

from src.explainability.shap_service import explain_transaction


# ---------------------------------------------------------
# 1. LOAD MODEL ONCE AND CACHE IT
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def get_model_bundle():
    """
    Load and cache the trained XGBoost model bundle.

    The model is loaded only once during the application
    lifecycle instead of once for every transaction.

    Returns:
        dict: Trained model pipeline and configuration.
    """
    return load_fraud_model()


# ---------------------------------------------------------
# 2. FRAUD DETECTION AGENT
# ---------------------------------------------------------

def fraud_detection_agent(state: FraudState) -> dict:
    """
    LangGraph node responsible for fraud detection.

    Steps:
        1. Read transaction from shared state.
        2. Load the trained XGBoost model.
        3. Predict fraud probability.
        4. Generate SHAP explanations.
        5. Return updates to shared state.

    Args:
        state (FraudState): Current LangGraph state.

    Returns:
        dict: Updated prediction, SHAP explanation and status.
    """

    # Step 1: Get transaction from shared state.
    transaction = state.get("transaction")

    if not transaction:
        raise ValueError(
            "Transaction data is missing from FraudState."
        )

    # Step 2: Load cached model bundle.
    model_bundle = get_model_bundle()

    # Step 3: Predict fraud risk using XGBoost.
    prediction = predict_transaction(
        transaction,
        model_bundle,
    )

    # Step 4: Generate SHAP explanation.
    shap_explanation = explain_transaction(
        transaction,
        model_bundle,
        top_n=5,
    )

    # Step 5: Return state updates.
    return {
        "prediction": prediction,
        "shap_explanation": shap_explanation,
        "status": "DETECTION_COMPLETED",
    }
