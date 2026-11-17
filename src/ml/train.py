import json
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from src.data.load_data import load_dataset

from src.preprocessing.preprocess import prepare_data


ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT_DIR / "models"

METRICS_DIR = ROOT_DIR / "reports" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

METRICS_DIR.mkdir(parents=True, exist_ok=True)


def find_best_threshold(y_true, probabilities):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        /
        (precision[:-1] + recall[:-1] + 1e-10)
    )

    best_index = f1_scores.argmax()

    return float(thresholds[best_index])


def main():

    # Load dataset

    df = load_dataset()

    df = load_dataset()
    print("Dataset loaded successfully!")

    # Prepare training, validation and testing data

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        preprocessor
    ) = prepare_data(df)

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Testing samples:", len(X_test))

    # Define ML models

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}

    best_model = None
    best_model_name = None
    best_ap = -1

    # Train and compare models

    for name, estimator in models.items():

        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", estimator)
        ])

        pipeline.fit(X_train, y_train)

        # Validation predictions

        val_probabilities = pipeline.predict_proba(
            X_val
        )[:, 1]

        ap = average_precision_score(
            y_val,
            val_probabilities
        )

        print("Validation PR-AUC:", round(ap, 5))

        results[name] = {
            "validation_pr_auc": float(ap)
        }

        if ap > best_ap:

            best_ap = ap

            best_model = pipeline

            best_model_name = name

    # Select threshold using validation data

    val_probabilities = best_model.predict_proba(
        X_val
    )[:, 1]

    best_threshold = find_best_threshold(
        y_val,
        val_probabilities
    )

    # Final testing

    test_probabilities = best_model.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    # Evaluation metrics

    final_ap = average_precision_score(
        y_test,
        test_probabilities
    )

    final_roc_auc = roc_auc_score(
        y_test,
        test_probabilities
    )

    report = classification_report(
        y_test,
        test_predictions,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        test_predictions
    )

    print("\nBest Model:", best_model_name)

    print("Selected Threshold:", best_threshold)

    print("Test PR-AUC:", final_ap)

    print("Test ROC-AUC:", final_roc_auc)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(matrix)

    # Save model and metadata

    model_bundle = {
        "pipeline": best_model,
        "model_name": best_model_name,
        "threshold": best_threshold,
        "feature_columns": list(X_train.columns)
    }

    joblib.dump(
        model_bundle,
        MODEL_DIR / "fraud_model.joblib"
    )

    metrics = {
        "selected_model": best_model_name,
        "validation_results": results,
        "threshold": best_threshold,
        "test_pr_auc": float(final_ap),
        "test_roc_auc": float(final_roc_auc),
        "classification_report": report,
        "confusion_matrix": matrix.tolist()
    }

    with open(
        METRICS_DIR / "evaluation.json",
        "w"
    ) as file:

        json.dump(metrics, file, indent=4)

    print("\nModel saved successfully!")

    print("Model path:", MODEL_DIR / "fraud_model.joblib")


if __name__ == "__main__":

    main()