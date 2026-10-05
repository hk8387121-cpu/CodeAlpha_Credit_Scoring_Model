from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "credit_data_processed.csv"
RESULTS_DIR = BASE_DIR / "results"

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError("Run 01_data_preprocessing.py first.")

    df = pd.read_csv(DATA_FILE)

    print("Dataset shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nSummary statistics:\n", df.describe().T)

    # Target distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="default_payment")
    plt.title("Credit Default Distribution")
    plt.xlabel("Default Payment (0 = No, 1 = Yes)")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "target_distribution.png", dpi=300)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(16, 12))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()

    # Credit amount distribution
    if "limit_bal" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x="limit_bal", hue="default_payment", bins=40, kde=True)
        plt.title("Credit Limit Distribution by Default Status")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "credit_limit_distribution.png", dpi=300)
        plt.close()

    # Age distribution
    if "age" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x="age", hue="default_payment", bins=30, kde=True)
        plt.title("Age Distribution by Default Status")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "age_distribution.png", dpi=300)
        plt.close()

    print("\nEDA completed. Figures saved in results/.")

if __name__ == "__main__":
    main()
