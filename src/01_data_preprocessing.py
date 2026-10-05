from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "credit_data_processed.csv"

def find_input_file():
    candidates = list(RAW_DIR.glob("*.xls")) + list(RAW_DIR.glob("*.xlsx")) + list(RAW_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No dataset found. Place the UCI dataset (.xls) inside data/raw/."
        )
    return candidates[0]

def load_dataset(path):
    if path.suffix.lower() in [".xls", ".xlsx"]:
        # The UCI file contains the header/variable-description rows used by this dataset.
        df = pd.read_excel(path, header=1)
    else:
        df = pd.read_csv(path)
    return df

def clean_column_names(df):
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df

def preprocess(df):
    df = df.copy()

    # Remove fully empty columns/rows.
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    # Normalize the UCI target column name.
    target_candidates = [
        "default_payment_next_month",
        "default.payment.next.month",
        "default_payment_next_month_",
    ]
    for col in target_candidates:
        if col in df.columns:
            df = df.rename(columns={col: "default_payment"})
            break

    if "default_payment" not in df.columns:
        raise ValueError(
            f"Target column not found. Available columns: {list(df.columns)}"
        )

    # The UCI dataset includes an ID column that is an identifier, not a predictive feature.
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Treat coded categorical variables as categorical values.
    categorical_cols = [
        c for c in ["sex", "education", "marriage"] if c in df.columns
    ]
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    # Replace common missing-value markers.
    df = df.replace(["?", "NA", "N/A", ""], np.nan)

    # Convert numeric-looking columns to numeric where possible.
    for col in df.columns:
        if col not in categorical_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows with missing target, then median-impute numeric feature values.
    df = df.dropna(subset=["default_payment"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_numeric_cols = [c for c in numeric_cols if c != "default_payment"]
    for col in feature_numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Convert target column to integer.
    df["default_payment"] = pd.to_numeric(
        df["default_payment"], errors="coerce"
).astype(int)

    df["default_payment"] = pd.to_numeric(df["default_payment"], errors="coerce").astype(int)

    return df

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    input_file = find_input_file()
    print(f"Loading: {input_file}")

    df = load_dataset(input_file)
    print(f"Original shape: {df.shape}")

    df = clean_column_names(df)
    df = preprocess(df)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Processed shape: {df.shape}")
    print("\nTarget distribution:")
    print(df["default_payment"].value_counts().sort_index())
    print(f"\nSaved processed dataset to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
