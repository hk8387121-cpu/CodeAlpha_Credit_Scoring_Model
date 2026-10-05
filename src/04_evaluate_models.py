from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "credit_data_processed.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RANDOM_STATE = 42
CATEGORICAL_FEATURES = ["sex", "education", "marriage"]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_features = [
        col for col in CATEGORICAL_FEATURES if col in X.columns
    ]
    numeric_features = [
        col for col in X.columns if col not in categorical_features
    ]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def get_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run src/01_data_preprocessing.py first."
        )

    df = pd.read_csv(DATA_FILE)
    X = df.drop(columns=["default_payment"])
    y = df["default_payment"].astype(int)

    # Recreate the same stratified 80/20 split used during training.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model_files = sorted(MODEL_DIR.glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError(
            "No trained models found. Run src/03_train_models.py first."
        )

    rows = []

    for model_file in model_files:
        name = model_file.stem
        model = joblib.load(model_file)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        rows.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "ROC_AUC": roc_auc,
        })

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Default", "Default"],
        )
        disp.plot()
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(
            RESULTS_DIR / f"confusion_matrix_{name}.png",
            dpi=300,
        )
        plt.close()

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc:.3f})",
        )
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            RESULTS_DIR / f"roc_curve_{name}.png",
            dpi=300,
        )
        plt.close()

    results = (
        pd.DataFrame(rows)
        .sort_values(
            by=["F1_Score", "ROC_AUC"],
            ascending=False,
        )
        .reset_index(drop=True)
    )
    results.to_csv(RESULTS_DIR / "model_results.csv", index=False)

    ax = results.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
    ].plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Credit Scoring Model Comparison")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=300)
    plt.close()

    best_model = results.iloc[0]["Model"]
    with open(RESULTS_DIR / "best_model.txt", "w", encoding="utf-8") as f:
        f.write(
            f"Best model based on F1-Score and ROC-AUC: {best_model}\n"
        )

    print("\nModel Evaluation Results:")
    print(results.to_string(index=False))
    print(f"\nBest model: {best_model}")
    print("\nEvaluation completed. Check results/.")


if __name__ == "__main__":
    main()
