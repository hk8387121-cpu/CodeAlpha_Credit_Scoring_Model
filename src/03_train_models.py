from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run src/01_data_preprocessing.py first."
        )

    df = pd.read_csv(DATA_FILE)

    if "default_payment" not in df.columns:
        raise ValueError("Target column 'default_payment' is missing.")

    X = df.drop(columns=["default_payment"])
    y = df["default_payment"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(X)

    models = {
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

    # Save only split metadata to the local results directory.
    # Customer-level test data is intentionally not required for the repository.
    split_info = pd.DataFrame([{
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_percentage": 80,
        "test_percentage": 20,
        "random_state": RANDOM_STATE,
    }])
    split_info.to_csv(RESULTS_DIR / "train_test_split.csv", index=False)

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)

        joblib.dump(
            pipeline,
            MODEL_DIR / f"{name}.joblib",
        )
        print(f"Saved: models/{name}.joblib")

        # Store test arrays in memory only; evaluation loads a reproducible split
        # using the same random state rather than committing customer-level data.

    print("\nAll models trained successfully.")


if __name__ == "__main__":
    main()
