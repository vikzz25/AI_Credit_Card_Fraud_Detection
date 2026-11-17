
"""
SHAP explanation service for fraud investigation.

Explains individual XGBoost predictions using SHAP values.
"""

import numpy as np
import pandas as pd
import shap


def explain_transaction(transaction, model_bundle, top_n=5):
    """
    Generate SHAP explanations for a single transaction.

    Args:
        transaction (dict): Transaction feature values.
        model_bundle (dict): Saved XGBoost model bundle.
        top_n (int): Number of important features to return.

    Returns:
        dict: Top contributing features and SHAP information.
    """

    pipeline = model_bundle["pipeline"]

    # Extract preprocessing and classifier components.
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    feature_columns = model_bundle["feature_columns"]

    # Prepare transaction in training feature order.
    transaction_df = pd.DataFrame(
        [[transaction[column] for column in feature_columns]],
        columns=feature_columns,
    )

    # Apply the same preprocessing used during training.
    transformed = preprocessor.transform(transaction_df)

    # Preserve original feature names.
    transformed_df = pd.DataFrame(
        transformed,
        columns=feature_columns,
    )

    # Create SHAP TreeExplainer.
    explainer = shap.TreeExplainer(classifier)

    # Calculate SHAP values.
    shap_values = explainer(transformed_df)

    # Extract contributions for the first transaction.
    values = np.asarray(shap_values.values)

    if values.ndim == 3:
        # Binary classification output: select Class 1.
        contributions = values[0, :, 1]
        base_value = np.asarray(shap_values.base_values)[0, 1]
    else:
        contributions = values[0]
        base_values = np.asarray(shap_values.base_values)
        base_value = float(base_values.reshape(-1)[0])

    # Rank features by absolute contribution.
    feature_importance = []

    for index, feature in enumerate(feature_columns):
        feature_importance.append({
            "feature": feature,
            "feature_value": float(
                transformed_df.iloc[0, index]
            ),
            "shap_value": float(contributions[index]),
            "direction": (
                "increases_model_output"
                if contributions[index] > 0
                else "decreases_model_output"
            ),
        })

    feature_importance.sort(
        key=lambda item: abs(item["shap_value"]),
        reverse=True,
    )

    return {
        "base_value": float(base_value),
        "top_features": feature_importance[:top_n],
        "explanation_method": "SHAP TreeExplainer",
        "output_note": (
            "SHAP contributions explain the model output. "
            "For XGBoost binary classification, these values "
            "are generally in raw margin (log-odds) space, "
            "not probability percentage points."
        ),
    }
