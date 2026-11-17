
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

from src.data.load_data import load_dataset


ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT_DIR / "models" / "fraud_model_xgboost.joblib"

FIGURE_DIR = ROOT_DIR / "reports" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def explain_model():

    # Load the trained model
    bundle = joblib.load(MODEL_PATH)

    pipeline = bundle["pipeline"]
    model_name = bundle["model_name"]

    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if model_name != "XGBoost":
        raise ValueError(
            "SHAP TreeExplainer in this script is configured for XGBoost. "
            f"The selected model is {model_name}."
        )

    print("Loaded model:", model_name)

    # Load data for explanations
    df = load_dataset()

    X = df.drop("Class", axis=1)

    # Use a reproducible sample to keep SHAP computation manageable
    X_sample = X.sample(
        n=min(1000, len(X)),
        random_state=42
    )

    # Apply the exact preprocessing fitted during training
    X_transformed = preprocessor.transform(X_sample)

    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()

    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=X_sample.index
    )

    # Create SHAP TreeExplainer
    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer.shap_values(
        X_transformed_df
    )

    # Global feature importance summary
    plt.figure()

    shap.summary_plot(
        shap_values,
        X_transformed_df,
        show=False,
        max_display=20
    )

    plt.title("SHAP Summary - Fraud Detection")

    summary_path = FIGURE_DIR / "shap_summary.png"

    plt.savefig(
        summary_path,
        bbox_inches="tight",
        dpi=300
    )

    plt.close()

    print("SHAP summary saved:", summary_path)

    # Global feature importance bar chart
    plt.figure()

    shap.summary_plot(
        shap_values,
        X_transformed_df,
        plot_type="bar",
        show=False,
        max_display=20
    )

    plt.title("SHAP Feature Importance")

    bar_path = FIGURE_DIR / "shap_feature_importance.png"

    plt.savefig(
        bar_path,
        bbox_inches="tight",
        dpi=300
    )

    plt.close()

    print("SHAP feature importance saved:", bar_path)

    # Explain one transaction
    single_transaction = X_transformed_df.iloc[[0]]

    single_shap_values = explainer.shap_values(
        single_transaction
    )

    plt.figure()

    shap.waterfall_plot(
        shap.Explanation(
            values=single_shap_values[0],
            base_values=explainer.expected_value,
            data=single_transaction.iloc[0].values,
            feature_names=feature_names
        ),
        show=False,
        max_display=15
    )

    waterfall_path = FIGURE_DIR / "shap_single_transaction.png"

    plt.savefig(
        waterfall_path,
        bbox_inches="tight",
        dpi=300
    )

    plt.close()

    print("Single transaction explanation saved:", waterfall_path)

    print("\nSHAP explanation completed!")


if __name__ == "__main__":
    explain_model()
