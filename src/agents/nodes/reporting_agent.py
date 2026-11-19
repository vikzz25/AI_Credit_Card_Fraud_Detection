"""
Reporting Agent.

Combines fraud prediction, SHAP explanations,
AI investigation and retrieved policy guidance
into a structured final investigation report.
"""

from src.agents.state.fraud_state import FraudState


# ---------------------------------------------------------
# 1. REPORTING AGENT
# ---------------------------------------------------------

def reporting_agent(state: FraudState) -> dict:
    """
    Generate a structured fraud investigation report.

    Args:
        state (FraudState): Current shared workflow state.

    Returns:
        dict: Final report and updated workflow status.
    """

    # -----------------------------------------------------
    # STEP 1: EXTRACT INFORMATION FROM SHARED STATE
    # -----------------------------------------------------

    prediction = state.get("prediction")
    shap_explanation = state.get("shap_explanation")
    investigation = state.get("investigation")
    policy_context = state.get("policy_context")

    # -----------------------------------------------------
    # STEP 2: VALIDATE REQUIRED INFORMATION
    # -----------------------------------------------------

    if not prediction:
        raise ValueError(
            "Prediction is missing from FraudState."
        )

    if not shap_explanation:
        raise ValueError(
            "SHAP explanation is missing from FraudState."
        )

    if not investigation:
        raise ValueError(
            "Investigation report is missing from FraudState."
        )

    if not policy_context:
        policy_context = (
            "No matching policy guidance was retrieved."
        )

    # -----------------------------------------------------
    # STEP 3: EXTRACT PREDICTION DETAILS
    # -----------------------------------------------------

    fraud_probability = prediction["fraud_probability"]

    prediction_details = {
        "fraud_probability": round(
            fraud_probability, 6
        ),
        "fraud_probability_percentage": round(
            fraud_probability * 100, 2
        ),
        "decision_threshold": prediction["threshold"],
        "prediction": prediction["prediction"],
        "risk_status": prediction["risk_status"],
    }

    # -----------------------------------------------------
    # STEP 4: EXTRACT SHAP DETAILS
    # -----------------------------------------------------

    shap_details = {
        "base_value": shap_explanation["base_value"],
        "top_features": shap_explanation["top_features"],
        "explanation_method": (
            shap_explanation["explanation_method"]
        ),
        "output_note": shap_explanation["output_note"],
    }

    # -----------------------------------------------------
    # STEP 5: DETERMINE INVESTIGATION SUMMARY
    # -----------------------------------------------------

    if prediction["prediction"] == 1:

        summary = (
            "The machine learning model classified "
            "the transaction as HIGH_RISK."
        )

        next_steps = [
            "Refer the transaction for review "
            "according to approved procedures.",
            "Examine available transaction evidence.",
            "Record the investigation outcome.",
        ]

    else:

        summary = (
            "The machine learning model classified "
            "the transaction as LOWER_RISK."
        )

        next_steps = [
            "Interpret the prediction alongside "
            "applicable monitoring procedures.",
            "Consider other available alerts or evidence.",
            "Record the assessment when required.",
        ]

    # -----------------------------------------------------
    # STEP 6: BUILD FINAL STRUCTURED REPORT
    # -----------------------------------------------------

    final_report = {

        "report_title": (
            "AI Credit Card Fraud Investigation Report"
        ),

        "executive_summary": summary,

        "risk_assessment": prediction_details,

        "model_explanation": shap_details,

        "ai_investigation": investigation,

        "policy_guidance": policy_context,

        "recommended_next_steps": next_steps,

        "limitations": [
            "A model prediction does not confirm fraud.",
            "SHAP explanations describe model behavior "
            "and do not establish causation.",
            "Anonymized features cannot be assigned "
            "unsupported real-world meanings.",
            "AI-generated recommendations are advisory.",
            "Final decisions require authorized human "
            "review according to applicable procedures.",
        ],

    }

    # -----------------------------------------------------
    # STEP 7: RETURN UPDATED WORKFLOW STATE
    # -----------------------------------------------------

    return {
        "final_report": final_report,
        "status": "REPORT_COMPLETED",
    }