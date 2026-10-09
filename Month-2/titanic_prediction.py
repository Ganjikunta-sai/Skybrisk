"""
Skybrisk Internship - Month 2, Task 2: Simple Predictive Model
Download the Titanic CSV from https://www.kaggle.com/datasets/yasserh/titanic-dataset
Save it beside this script as Titanic-Dataset.csv.
Install: python -m pip install pandas scikit-learn
Run: python titanic_prediction.py
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = Path(__file__).parent / "Titanic-Dataset.csv"
TARGET = "Survived"

def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Download the Titanic CSV and save it as Titanic-Dataset.csv beside this script.")
        print("Dataset: https://www.kaggle.com/datasets/yasserh/titanic-dataset")
        return

    df = pd.read_csv(DATA_PATH)
    print("--- First 5 rows ---")
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nMissing values:\n", df.isnull().sum())

    if TARGET not in df.columns:
        print(f"Target column '{TARGET}' not found. Please check your CSV.")
        return

    drop_cols = [c for c in ["PassengerId", "Name", "Ticket", "Cabin"] if c in df.columns]
    X = df.drop(columns=[TARGET] + drop_cols)
    y = df[TARGET]
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid].astype(int)

    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("preprocessing", preprocess),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\n--- Evaluation ---")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
    print("\nClassification report:\n",
          classification_report(y_test, pred, zero_division=0))
    print("Use the actual results above in your report; do not guess the score.")

if __name__ == "__main__":
    main()
