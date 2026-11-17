
import json
import joblib

from pathlib import Path

from sklearn.base import clone
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

from xgboost import XGBClassifier

from src.data.load_data import load_dataset
from src.preprocessing.preprocess import prepare_data


ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "reports" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def find_best_threshold(y_true, probabilities):
    """Find the classification threshold with the highest validation F1."""

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    best_index = f1_scores.argmax()

    return float(thresholds[best_index])


def main():

    print("Loading dataset...")

    df = load_dataset()

    print("Dataset loaded:", df.shape)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        preprocessor
    ) = prepare_data(df)

    print("\nTraining samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Testing samples:", len(X_test))

    # Calculate class imbalance using training data only
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = negative_count / positive_count

    print("\nGenuine training transactions:", negative_count)
    print("Fraud training transactions:", positive_count)
    print("Scale positive weight:", round(scale_pos_weight, 2))

    # Define models
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
        ),

        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=42,
            n_jobs=4
        )
    }

    results = {}

    trained_models = {}

    best_model = None
    best_model_name = None
    best_ap = -1
    best_threshold = 0.5

    # Train and compare models
    for name, estimator in models.items():

        print(f"\n{'=' * 50}")
        print(f"Training {name}")
        print(f"{'=' * 50}")

        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", estimator)
        ])

        pipeline.fit(X_train, y_train)

        trained_models[name] = pipeline

        val_probabilities = pipeline.predict_proba(X_val)[:, 1]

        validation_ap = average_precision_score(
            y_val,
            val_probabilities
        )

        threshold = find_best_threshold(
            y_val,
            val_probabilities
        )

        val_predictions = (
            val_probabilities >= threshold
        ).astype(int)

        print("Validation Average Precision:", round(validation_ap, 5))
        print("Validation F1-optimal threshold:", round(threshold, 5))

        results[name] = {
            "validation_average_precision": float(validation_ap),
            "validation_threshold": float(threshold),
            "validation_classification_report": classification_report(
                y_val,
                val_predictions,
                output_dict=True,
                zero_division=0
            )
        }

        if validation_ap > best_ap:

            best_ap = validation_ap
            best_model = pipeline
            best_model_name = name
            best_threshold = threshold

    # Save the dedicated XGBoost model for SHAP and Phase 3.

    xgboost_bundle = {
        "pipeline": trained_models["XGBoost"],
        "model_name": "XGBoost",
        "threshold": results["XGBoost"]["validation_threshold"],
        "feature_columns": list(X_train.columns),
    }

    joblib.dump(
        xgboost_bundle,
        MODEL_DIR / "fraud_model_xgboost.joblib"
    )

    print("XGBoost model saved successfully.")

    # Save XGBoost separately for SHAP explanations
    xgboost_bundle = {
    "pipeline": trained_models["XGBoost"],
    "model_name": "XGBoost",
    "threshold": results["XGBoost"]["validation_threshold"],
    "feature_columns": list(X_train.columns)
    }

    joblib.dump(
    xgboost_bundle,
    MODEL_DIR / "fraud_model_xgboost.joblib"
    )

    print("XGBoost model saved separately for SHAP.")

    # Evaluate selected model on the untouched test set
    print("\nSelected model:", best_model_name)

    test_probabilities = best_model.predict_proba(X_test)[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    test_ap = average_precision_score(
        y_test,
        test_probabilities
    )

    test_roc_auc = roc_auc_score(
        y_test,
        test_probabilities
    )

    test_report = classification_report(
        y_test,
        test_predictions,
        output_dict=True,
        zero_division=0
    )

    test_matrix = confusion_matrix(
        y_test,
        test_predictions
    )

    print("\nTest Average Precision:", round(test_ap, 5))
    print("Test ROC-AUC:", round(test_roc_auc, 5))
    print("Selected threshold:", round(best_threshold, 5))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(test_matrix)

    # Save selected model and metadata
    model_bundle = {
        "pipeline": best_model,
        "model_name": best_model_name,
        "threshold": best_threshold,
        "feature_columns": list(X_train.columns)
    }

    model_path = MODEL_DIR / "fraud_model_advanced.joblib"

    joblib.dump(model_bundle, model_path)

    # Save metrics
    metrics = {
        "selected_model": best_model_name,
        "validation_results": results,
        "selected_threshold": float(best_threshold),
        "test_average_precision": float(test_ap),
        "test_roc_auc": float(test_roc_auc),
        "test_classification_report": test_report,
        "test_confusion_matrix": test_matrix.tolist()
    }

    metrics_path = METRICS_DIR / "advanced_evaluation.json"

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nTraining completed successfully!")
    print("Model saved:", model_path)
    print("Metrics saved:", metrics_path)


if __name__ == "__main__":
    main()
