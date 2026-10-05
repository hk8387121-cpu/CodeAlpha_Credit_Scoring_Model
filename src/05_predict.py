from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

def main():
    model_path = MODEL_DIR / "random_forest.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "Random Forest model not found. Run 03_train_models.py first."
        )

    model = joblib.load(model_path)

    # Example customer. Replace values with a real test record from your dataset.
    # The column names must match the processed dataset.
    sample = pd.DataFrame([{
        "limit_bal": 200000,
        "sex": 2,
        "education": 2,
        "marriage": 1,
        "age": 35,
        "pay_0": 0,
        "pay_2": 0,
        "pay_3": 0,
        "pay_4": 0,
        "pay_5": 0,
        "pay_6": 0,
        "bill_amt1": 50000,
        "bill_amt2": 48000,
        "bill_amt3": 45000,
        "bill_amt4": 42000,
        "bill_amt5": 40000,
        "bill_amt6": 38000,
        "pay_amt1": 5000,
        "pay_amt2": 5000,
        "pay_amt3": 5000,
        "pay_amt4": 5000,
        "pay_amt5": 5000,
        "pay_amt6": 5000
    }])

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0, 1]

    label = "High Risk / Default" if prediction == 1 else "Lower Risk / No Default"

    print("Prediction:", label)
    print(f"Default probability: {probability:.2%}")

if __name__ == "__main__":
    main()
