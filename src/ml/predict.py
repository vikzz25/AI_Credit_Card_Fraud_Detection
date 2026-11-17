import joblib

import pandas as pd

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT_DIR / "models" / "fraud_model.joblib"


def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Train the model before making predictions."
        )

    return joblib.load(MODEL_PATH)


def predict_transaction(transaction):

    bundle = load_model()

    model = bundle["pipeline"]

    threshold = bundle["threshold"]

    feature_columns = bundle["feature_columns"]

    # Validate input features

    missing_features = [
        column
        for column in feature_columns
        if column not in transaction
    ]

    if missing_features:

        raise ValueError(
            f"Missing features: {missing_features}"
        )

    # Convert input into DataFrame

    transaction_df = pd.DataFrame(
        [transaction],
        columns=feature_columns
    )

    # Get fraud probability

    fraud_probability = model.predict_proba(
        transaction_df
    )[0][1]

    # Apply selected threshold

    prediction = int(
        fraud_probability >= threshold
    )

    return {

        "prediction": prediction,

        "fraud_probability": round(
            float(fraud_probability), 4
        ),

        "threshold": round(
            float(threshold), 4
        ),

        "status": (
            "Potential Fraud"
            if prediction == 1
            else "Likely Genuine"
        )

    }