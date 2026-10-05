from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_file = RESULTS_DIR / "test_data.csv"
    if not test_file.exists():
        raise FileNotFoundError("Run 03_train_models.py first.")

    test_df = pd.read_csv(test_file)
    y_test = test_df["default_payment"]
    X_test = test_df.drop(columns=["default_payment"])

    model_files = sorted(MODEL_DIR.glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError("No trained models found in models/.")

    rows = []
    plt.figure(figsize=(8, 6))

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
            "ROC_AUC": roc_auc
        })

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Default", "Default"]
        )
        disp.plot()
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"confusion_matrix_{name}.png", dpi=300)
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"roc_curve_{name}.png", dpi=300)
        plt.close()

    results = pd.DataFrame(rows).sort_values(
        by=["F1_Score", "ROC_AUC"],
        ascending=False
    )
    results.to_csv(RESULTS_DIR / "model_results.csv", index=False)

    # Model comparison chart
    metrics = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
    ax = results.set_index("Model")[metrics].plot(
        kind="bar", figsize=(12, 6)
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Credit Scoring Model Comparison")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=300)
    plt.close()

    best_model = results.iloc[0]["Model"]
    with open(RESULTS_DIR / "best_model.txt", "w", encoding="utf-8") as f:
        f.write(f"Best model based on F1-Score and ROC-AUC: {best_model}\n")

    print("\nModel Evaluation Results:")
    print(results.to_string(index=False))
    print(f"\nBest model: {best_model}")
    print("\nEvaluation completed. Check results/.")

if __name__ == "__main__":
    main()
