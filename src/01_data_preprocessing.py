from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "credit_data_processed.csv"

CATEGORICAL_COLUMNS = ["sex", "education", "marriage"]
TARGET_COLUMN = "default_payment"


def find_input_file():
    candidates = (
        list(RAW_DIR.glob("*.xls"))
        + list(RAW_DIR.glob("*.xlsx"))
        + list(RAW_DIR.glob("*.csv"))
    )
    if not candidates:
        raise FileNotFoundError(
            "No dataset found. Place the UCI dataset (.xls) inside data/raw/."
        )
    return candidates[0]


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xls", ".xlsx"]:
        return pd.read_excel(path, header=1)
    return pd.read_csv(path)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    ]
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove fully empty rows and columns.
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    # Normalize the UCI target name.
    target_candidates = [
        "default_payment_next_month",
        "default.payment.next.month",
        "default_payment_next_month_",
    ]
    for col in target_candidates:
        if col in df.columns:
            df = df.rename(columns={col: TARGET_COLUMN})
            break

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column not found. Available columns: {list(df.columns)}"
        )

    # ID is an identifier and is not used as a predictive feature.
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Preserve categorical columns as object values for downstream one-hot encoding.
    categorical_cols = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
    df = df.replace(["?", "NA", "N/A", ""], np.nan)

    # Convert non-categorical feature columns to numeric.
    for col in df.columns:
        if col not in categorical_cols and col != TARGET_COLUMN:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="coerce"
    )
    df = df.dropna(subset=[TARGET_COLUMN])

    # Impute numerical feature values using the training-independent dataset median
    # at this preprocessing stage; the model pipeline also contains safe imputers.
    numeric_feature_cols = [
        col for col in df.columns
        if col not in categorical_cols + [TARGET_COLUMN]
    ]
    for col in numeric_feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

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
    print(df[TARGET_COLUMN].value_counts().sort_index())
    print(f"\nSaved processed dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
