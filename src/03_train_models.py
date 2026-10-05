from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "credit_data_processed.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RANDOM_STATE = 42

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError("Run 01_data_preprocessing.py first.")

    df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=["default_payment"])
    y = df["default_payment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

        # Categorical features
    categorical_features = [
        col for col in ["sex", "education", "marriage"]
        if col in X.columns
    ]

    # All remaining features are treated as numerical.
    numeric_features = [
        col for col in X.columns
        if col not in categorical_features
    ]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=10,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    split_info = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_percentage": 80,
        "test_percentage": 20
    }
    pd.DataFrame([split_info]).to_csv(RESULTS_DIR / "train_test_split.csv", index=False)

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)

        joblib.dump(pipeline, MODEL_DIR / f"{name}.joblib")
        print(f"Saved: models/{name}.joblib")

    # Save test data for reproducible evaluation.
    test_df = X_test.copy()
    test_df["default_payment"] = y_test.values
    test_df.to_csv(RESULTS_DIR / "test_data.csv", index=False)

    print("\nAll models trained successfully.")

if __name__ == "__main__":
    main()
