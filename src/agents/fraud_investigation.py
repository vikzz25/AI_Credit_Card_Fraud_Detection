
"""
Fraud Investigation Assistant.

Uses the trained XGBoost pipeline to predict transaction fraud risk
and LangChain + OpenAI to generate an evidence-based explanation.

The ML model is responsible for the prediction.
The LLM is responsible only for explaining the supplied evidence.
"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.explainability.shap_service import explain_transaction
from src.data.load_data import load_dataset


# ---------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fraud_model_xgboost.joblib"
)

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


# ---------------------------------------------------------
# 2. LOAD TRAINED XGBOOST MODEL
# ---------------------------------------------------------

def load_fraud_model():
    """
    Load the saved XGBoost model bundle.

    Returns:
        dict: Model pipeline, threshold and feature information.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost model not found: {MODEL_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    required_keys = {
        "pipeline",
        "threshold",
        "feature_columns",
    }

    missing_keys = required_keys - set(bundle.keys())

    if missing_keys:
        raise ValueError(
            f"Model bundle is missing keys: {missing_keys}"
        )

    return bundle


# ---------------------------------------------------------
# 3. PREDICT FRAUD RISK
# ---------------------------------------------------------

def predict_transaction(transaction, model_bundle):
    """
    Predict the fraud probability of one transaction.

    Args:
        transaction (dict): Transaction feature values.
        model_bundle (dict): Saved XGBoost model bundle.

    Returns:
        dict: Prediction, probability and applied threshold.
    """

    feature_columns = model_bundle["feature_columns"]
    threshold = float(model_bundle["threshold"])
    pipeline = model_bundle["pipeline"]

    # Ensure the input contains all expected model features.
    missing_features = [
        column
        for column in feature_columns
        if column not in transaction
    ]

    if missing_features:
        raise ValueError(
            f"Missing transaction features: {missing_features}"
        )

    # Keep the same feature order used during training.
    transaction_df = pd.DataFrame(
        [[transaction[column] for column in feature_columns]],
        columns=feature_columns,
    )

    # Get probability of Class 1 (fraud).
    fraud_probability = float(
        pipeline.predict_proba(transaction_df)[0][1]
    )

    # Apply the threshold selected during Phase 2.
    prediction = int(fraud_probability >= threshold)

    return {
        "fraud_probability": fraud_probability,
        "threshold": threshold,
        "prediction": prediction,
        "risk_status": (
            "HIGH_RISK"
            if prediction == 1
            else "LOWER_RISK"
        ),
    }


# ---------------------------------------------------------
# 4. INITIALIZE OPENAI MODEL
# ---------------------------------------------------------

def create_llm():
    """
    Initialize the OpenAI chat model through LangChain.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    model_name = os.getenv(
        "OPENAI_MODEL",
        "gpt-4.1-mini",
    )

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Check your .env file."
        )

    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key,
    )


# ---------------------------------------------------------
# 5. CREATE INVESTIGATION PROMPT
# ---------------------------------------------------------

def create_investigation_prompt():
    """
    Create instructions for the AI investigation assistant.
    """


    system_message = """
    You are an AI assistant supporting credit card fraud investigation.

    Your responsibility is to interpret the supplied machine learning
    prediction and supporting evidence.

    STRICT RULES:

    1. The supplied XGBoost prediction, fraud probability and
   classification threshold are authoritative.
   Never modify or recalculate them.

    2. Do not invent transaction details, cardholder identities,
   merchant names, locations or purchase categories.

    3. Features V1 to V28 are anonymized PCA-transformed numerical
   features. Their individual real-world meanings are unknown.

    4. The Time feature represents elapsed seconds since the first
   transaction in the dataset. It is not a clock time.

    5. Distinguish between a model prediction and confirmed fraud.
   A high-risk prediction does not prove fraud.

    6. Never recommend automatically blocking a card or rejecting
   a transaction. Recommendations must remain advisory.

    7. Explain uncertainty and recommend appropriate human review
   in accordance with established fraud-monitoring policies.

    8. If evidence is insufficient, clearly state this limitation.

    IMPORTANT: UNDERSTANDING THE NUMERICAL EVIDENCE

    - Fraud probability is expressed as a probability between 0 and 1.
    - The decision threshold is also a probability threshold.
    - The prediction is determined by comparing fraud probability
  against that threshold.
    - SHAP contributions for this XGBoost model are generally in
  raw-margin (log-odds) space, not probability percentage points.
    - Positive SHAP values increase the model's raw output.
    - Negative SHAP values decrease the model's raw output.
    - The base value represents the baseline model output.
    - Only the supplied top SHAP features are available. Do not
      assume their contributions represent the complete explanation.
    - Do not claim that the top five features alone account for
      the entire difference between the baseline and prediction.
    - Never interpret SHAP contributions as percentages.

    REPORT STRUCTURE:

    Use these sections:

    1. Risk assessment
    2. Model prediction
    3. Available transaction evidence
    4. Explanation and limitations
    5. Recommended next steps

    Keep the report professional, understandable and evidence-based.
    Use readable spacing and bullet points.
    """

    human_message = """
    Investigate the following credit card transaction using only
    the evidence provided below.

    TRANSACTION INFORMATION:
    {transaction_data}

    XGBOOST PREDICTION:
    {prediction_data}

    SHAP EXPLANATION:
    {shap_data}

    Your investigation must:

    1. Explain the model's risk classification and fraud probability.
    2. Compare the probability with the supplied decision threshold.
    3. Identify the most influential supplied SHAP features.
    4. Explain whether each feature increases or decreases the
       model's raw output.
    5. Clarify that V1 to V28 are anonymized features.
    6. Explain that SHAP contributions are not probability percentages.
    7. Avoid claiming that the top five features fully explain
       the model's prediction.
    8. Recommend appropriate monitoring or human review without
       making autonomous financial decisions.

    Do not invent evidence or change any supplied numerical values.

    Generate a structured investigation report.
    """

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", human_message),
        ]
    )


# ---------------------------------------------------------
# 6. GENERATE INVESTIGATION REPORT
# ---------------------------------------------------------

def investigate_transaction(transaction, model_bundle, llm):
    """
    Predict, explain and investigate one transaction.

    Args:
        transaction (dict): Transaction feature values.
        model_bundle (dict): Loaded XGBoost model bundle.
        llm: Initialized LangChain chat model.

    Returns:
        dict: Prediction, SHAP explanation and AI report.
    """

    # Step A: Get ML prediction.
    prediction_result = predict_transaction(
        transaction,
        model_bundle,
    )

    # Step B: Generate SHAP explanations.
    shap_result = explain_transaction(
        transaction,
        model_bundle,
        top_n=5,
    )

    # Step C: Prepare transaction evidence.
    transaction_evidence = {
        key: round(float(value), 6)
        for key, value in transaction.items()
    }

    # Step D: Prepare prediction evidence.
    prediction_evidence = {
        "fraud_probability_percentage": round(
            prediction_result["fraud_probability"] * 100,
            2,
        ),
        "decision_threshold": round(
            prediction_result["threshold"],
            6,
        ),
        "prediction": prediction_result["prediction"],
        "risk_status": prediction_result["risk_status"],
    }

    # Step E: Create combined investigation evidence.
    investigation_evidence = {
        "shap_base_value": shap_result["base_value"],
        "top_features": shap_result["top_features"],
        "explanation_method": shap_result["explanation_method"],
        "output_note": shap_result["output_note"],
    }

    # Step F: Build LangChain prompt.
    prompt = create_investigation_prompt()

    # Add SHAP evidence to the prompt.
    prompt = prompt.partial(
        shap_data=json.dumps(
            investigation_evidence,
            indent=2,
        )
    )

    chain = prompt | llm

    # Step G: Generate AI report.
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

    return {
        "prediction": prediction_result,
        "shap_explanation": shap_result,
        "investigation_report": response.content,
    }


# ---------------------------------------------------------
# 7. RUN GENUINE AND FRAUDULENT SAMPLE INVESTIGATIONS
# ---------------------------------------------------------

def main():
    """
    Run investigations on genuine and fraudulent transactions.
    """

    # Step 1: Load XGBoost model.
    print("\nLoading XGBoost model...")

    model_bundle = load_fraud_model()

    print("XGBoost model loaded successfully.")

    # Step 2: Load dataset.
    print("\nLoading dataset...")

    df = load_dataset()

    if "Class" not in df.columns:
        raise ValueError(
            "Dataset does not contain Class column."
        )

    feature_columns = model_bundle["feature_columns"]

    if "Class" in feature_columns:
        raise ValueError(
            "Class must not be included as a model input feature."
        )

    # Step 3: Validate dataset features.
    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing features: {missing_columns}"
        )

    # Step 4: Select genuine and fraudulent transactions.
    genuine_samples = df[df["Class"] == 0]
    fraud_samples = df[df["Class"] == 1]

    if genuine_samples.empty or fraud_samples.empty:
        raise ValueError(
            "Both genuine and fraudulent transactions must be available."
        )

    test_cases = [
        ("GENUINE", genuine_samples.iloc[0]),
        ("FRAUDULENT", fraud_samples.iloc[0]),
    ]

    # Step 5: Initialize OpenAI.
    print("\nInitializing OpenAI model...")

    llm = create_llm()

    print("OpenAI model initialized.")

    # Step 6: Investigate both transactions.
    for actual_label, sample in test_cases:

        transaction = {
            column: float(sample[column])
            for column in feature_columns
        }

        print("\n" + "=" * 60)
        print(f"TESTING {actual_label} TRANSACTION")
        print("=" * 60)

        result = investigate_transaction(
            transaction,
            model_bundle,
            llm,
        )

        print("\nActual dataset label:", actual_label)

        print(
            "Predicted risk:",
            result["prediction"]["risk_status"],
        )

        print(
            "Fraud probability:",
            f"{result['prediction']['fraud_probability'] * 100:.2f}%"
        )

        print(
            "Decision threshold:",
            result["prediction"]["threshold"],
        )

        # Display SHAP feature contributions.
        print("\nSHAP FEATURE CONTRIBUTIONS")
        print("-" * 60)

        for feature in result["shap_explanation"]["top_features"]:

            print(
                f"{feature['feature']}: "
                f"SHAP = {feature['shap_value']:.6f}, "
                f"Direction = {feature['direction']}"
            )

        # Display AI investigation report.
        print("\nAI INVESTIGATION REPORT")
        print("=" * 60)

        print(result["investigation_report"])


# ---------------------------------------------------------
# 8. APPLICATION ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
