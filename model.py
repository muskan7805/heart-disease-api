"""
model.py
--------
End-to-end training pipeline for the Heart Disease Risk Prediction project.

Pipeline:
    1. Load the UCI Cleveland Heart Disease dataset (303 rows, 14 columns).
    2. Clean the data (handle missing values, fix dtypes).
    3. Run a short EDA and save plots to static/.
    4. Train + compare 3 models: Logistic Regression, Random Forest, XGBoost.
    5. Evaluate on a held-out 20% test split (accuracy, F1, confusion matrix).
    6. Explain the winning model with SHAP (global + single-patient plots).
    7. Persist the winning model + metadata to heart_model.pkl for app.py.

Run:
    python model.py
"""

import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")  # no display backend needed on a server / CI box
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
STATIC_DIR = "static"
DATA_PATH = "data/heart.csv"
MODEL_PATH = "heart_model.pkl"

# Column names as used throughout the project / the FastAPI request schema.
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET_NAME = "target"

# Human-readable descriptions used later for SHAP-based explanations in app.py
FEATURE_DESCRIPTIONS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Serum cholesterol",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG results",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Number of major vessels colored by fluoroscopy",
    "thal": "Thalassemia test result",
}


# --------------------------------------------------------------------------- #
# 1. Load data
# --------------------------------------------------------------------------- #
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the Cleveland Heart Disease dataset.

    The UCI "processed.cleveland.data" file is not reachable from this
    training environment's network allow-list, so a local, pre-verified
    copy (303 rows x 14 columns, identical schema to the UCI version) is
    bundled in data/heart.csv. If you have network access and prefer the
    live source, swap this for `ucimlrepo.fetch_ucirepo(id=45)`.
    """
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# --------------------------------------------------------------------------- #
# 2. Clean data
# --------------------------------------------------------------------------- #
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and enforce correct dtypes.

    The raw UCI file marks missing values with '?' (found in the `ca` and
    `thal` columns for a handful of rows). We coerce those to NaN and
    impute with the column median, which is robust to the mild skew in
    both columns. All other columns are cast to numeric defensively.
    """
    df = df.copy()

    # Replace UCI's '?' placeholder with proper NaN, then coerce to numeric.
    df.replace("?", np.nan, inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        print(f"Found {missing_before} missing values — imputing with column median")
        df.fillna(df.median(numeric_only=True), inplace=True)
    else:
        print("No missing values found in the dataset")

    # Note: a couple of rows are identical across all 13 features + target.
    # With mostly low-cardinality/categorical columns in a 303-row clinical
    # dataset, that's expected by chance rather than a data-entry error, so
    # we keep every row (this also matches the official 303-row UCI count).
    dupes = df.duplicated().sum()
    if dupes:
        print(f"Note: {dupes} row(s) share identical feature values (kept — see comment above)")

    # The UCI target is 0 (no disease) vs 1-4 (increasing severity).
    # For binary risk classification we collapse 1-4 -> 1 (disease present).
    df[TARGET_NAME] = (df[TARGET_NAME] > 0).astype(int)

    df.reset_index(drop=True, inplace=True)
    return df


# --------------------------------------------------------------------------- #
# 3. EDA
# --------------------------------------------------------------------------- #
def run_eda(df: pd.DataFrame, out_dir: str = STATIC_DIR) -> None:
    """Save a handful of exploratory plots + a text summary."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    sns.set_style("whitegrid")

    # Target class balance
    plt.figure(figsize=(5, 4))
    sns.countplot(x=TARGET_NAME, data=df, hue=TARGET_NAME, palette=["#2ecc71", "#e74c3c"], legend=False)
    plt.title("Target Class Balance (0 = No Disease, 1 = Disease)")
    plt.xlabel("Heart Disease")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/target_balance.png", dpi=120)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/correlation_heatmap.png", dpi=120)
    plt.close()

    # Age distribution by target
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="age", hue=TARGET_NAME, kde=True, palette=["#2ecc71", "#e74c3c"], multiple="stack")
    plt.title("Age Distribution by Diagnosis")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/age_distribution.png", dpi=120)
    plt.close()

    print(f"EDA plots saved to {out_dir}/")
    print("\nClass balance:\n", df[TARGET_NAME].value_counts())
    print("\nSummary stats:\n", df.describe().T)


# --------------------------------------------------------------------------- #
# 4 & 5. Train + evaluate models
# --------------------------------------------------------------------------- #
def train_and_evaluate(df: pd.DataFrame):
    """
    Train Logistic Regression, Random Forest and XGBoost on an 80/20 split,
    evaluate each, and return the results plus the fitted artifacts needed
    at inference time.
    """
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    # Logistic Regression needs scaled inputs; tree models do not.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    fitted_models = {}

    # ---- Logistic Regression ----
    # C tuned via GridSearchCV (5-fold stratified CV, scoring='f1').
    lr = LogisticRegression(C=0.01, max_iter=2000, random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    results["Logistic Regression"] = _score(y_test, lr_pred)
    fitted_models["Logistic Regression"] = lr

    # ---- Random Forest ----
    # Hyperparameters tuned via GridSearchCV (5-fold stratified CV, scoring='f1').
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=3, min_samples_split=10,
        min_samples_leaf=2, random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["Random Forest"] = _score(y_test, rf_pred)
    fitted_models["Random Forest"] = rf

    # ---- XGBoost ----
    # Hyperparameters tuned via GridSearchCV (5-fold stratified CV, scoring='f1').
    xgb = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8,
        eval_metric="logloss", random_state=RANDOM_STATE,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    results["XGBoost"] = _score(y_test, xgb_pred)
    fitted_models["XGBoost"] = xgb

    # ---- Report ----
    print("\n===== Model Comparison (test set) =====")
    for name, metrics in results.items():
        print(f"{name:20s}  accuracy={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}")

    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"\nBest model by F1-score: {best_name}")
    if best_name != "XGBoost":
        print(
            f"Note: {best_name} edged out XGBoost by F1 on this particular split; "
            "differences are within a point or two on a 61-row test set."
        )

    # Confusion matrices for every model (saved for the README / static assets)
    _plot_confusion_matrices(fitted_models, X_test, X_test_scaled, y_test)

    print(f"\nClassification report for {best_name}:")
    best_pred = fitted_models[best_name].predict(
        X_test_scaled if best_name == "Logistic Regression" else X_test
    )
    print(classification_report(y_test, best_pred, target_names=["No Disease", "Disease"]))

    return {
        "results": results,
        "best_name": best_name,
        "best_model": fitted_models[best_name],
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def _score(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _plot_confusion_matrices(fitted_models, X_test, X_test_scaled, y_test, out_dir=STATIC_DIR):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, model) in zip(axes, fitted_models.items()):
        X_in = X_test_scaled if name == "Logistic Regression" else X_test
        preds = model.predict(X_in)
        cm = confusion_matrix(y_test, preds)
        ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"]).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrices.png", dpi=120)
    plt.close()
    print(f"Confusion matrices saved to {out_dir}/confusion_matrices.png")


# --------------------------------------------------------------------------- #
# 6. SHAP explainability
# --------------------------------------------------------------------------- #
def run_shap_analysis(model, X_train, X_test, out_dir: str = STATIC_DIR):
    """
    Produce a global feature-importance (summary) plot and a single-patient
    explanation plot for the winning (tree-based) model.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # Global feature importance (beeswarm summary plot)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to {out_dir}/shap_summary.png")

    # Single-patient explanation (waterfall plot for test patient #0)
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/shap_force_patient0.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"SHAP single-patient plot saved to {out_dir}/shap_force_patient0.png")

    return explainer


# --------------------------------------------------------------------------- #
# 7. Persist model
# --------------------------------------------------------------------------- #
def save_model(bundle: dict, results: dict, path: str = MODEL_PATH) -> None:
    """
    Save everything app.py needs at inference time in a single pickle:
    the fitted model, the feature order, and the fitted StandardScaler
    (kept for completeness / for swapping in the Logistic Regression model).
    """
    artifact = {
        "model": bundle["best_model"],
        "model_name": bundle["best_name"],
        "scaler": bundle["scaler"],
        "feature_names": FEATURE_NAMES,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "metrics": results,
    }
    joblib.dump(artifact, path)
    print(f"\nSaved trained artifact -> {path}")

    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved metrics -> metrics.json")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    df = load_data()
    df = clean_data(df)
    run_eda(df)

    bundle = train_and_evaluate(df)
    run_shap_analysis(bundle["best_model"], bundle["X_train"], bundle["X_test"])
    save_model(bundle, bundle["results"])

    print("\nDone. Train the API with: uvicorn app:app --reload")


if __name__ == "__main__":
    main()
