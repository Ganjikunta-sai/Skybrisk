"""
Month 5: Flight Delay Prediction Using Machine Learning
Place your Airline Delay CSV in Month-5/data/ and update DATA_PATH below.

Because Kaggle airline-delay datasets use different column names and definitions,
check the CSV headers and set TARGET_COLUMN / column mappings as needed.
This starter treats a flight as delayed when the target column already contains
a binary 0/1 label, or when the target is a numeric delay in minutes.
"""
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "airline_delay.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Change this to match your dataset.
TARGET_COLUMN = "ArrDel15"  # Common in some flight datasets; inspect your CSV first.
DELAY_MINUTES_COLUMN = None  # e.g. "ArrDelay"; use only if no binary target exists.
DATE_COLUMN_CANDIDATES = ["FlightDate", "Flight_Date", "flight_date", "Date"]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Create Month-5/data/ and save your CSV as airline_delay.csv."
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())

    # Remove columns with extremely high missingness (>60%).
    missing_ratio = df.isna().mean()
    drop_missing_cols = missing_ratio[missing_ratio > 0.60].index.tolist()
    df = df.drop(columns=drop_missing_cols)
    print("Dropped columns with >60% missing:", drop_missing_cols)

    # Create date features if a date column is available.
    date_col = next((c for c in DATE_COLUMN_CANDIDATES if c in df.columns), None)
    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        df["FlightWeekday"] = dates.dt.day_name()
        df["FlightMonth"] = dates.dt.month
        df = df.drop(columns=[date_col])

    # Build target. Prefer an existing binary delay label; otherwise use delay minutes.
    if TARGET_COLUMN in df.columns:
        target = df[TARGET_COLUMN]
        if target.dropna().nunique() > 2:
            warnings.warn(
                f"{TARGET_COLUMN} has more than 2 values; interpreting positive values as delayed."
            )
            y = (pd.to_numeric(target, errors="coerce") > 0).astype("Int64")
        else:
            # Handles numeric 0/1 and common string yes/no labels.
            numeric_target = pd.to_numeric(target, errors="coerce")
            if numeric_target.notna().mean() > 0.8:
                y = numeric_target
            else:
                normalized = target.astype(str).str.strip().str.lower()
                mapping = {"yes": 1, "no": 0, "delayed": 1, "on time": 0,
                           "on-time": 0, "true": 1, "false": 0}
                y = normalized.map(mapping)
        df = df.drop(columns=[TARGET_COLUMN])
    elif DELAY_MINUTES_COLUMN and DELAY_MINUTES_COLUMN in df.columns:
        delay_minutes = pd.to_numeric(df[DELAY_MINUTES_COLUMN], errors="coerce")
        y = (delay_minutes > 15).astype("Int64")
        df = df.drop(columns=[DELAY_MINUTES_COLUMN])
    else:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found. "
            f"Available columns: {df.columns.tolist()}\n"
            "Edit TARGET_COLUMN near the top of this script to match a binary delay label. "
            "Or set DELAY_MINUTES_COLUMN to a numeric delay-in-minutes column."
        )

    # Remove rows without a valid target.
    y = pd.Series(y, index=df.index)
    valid = y.notna()
    df, y = df.loc[valid].copy(), y.loc[valid].astype(int)
    if y.nunique() != 2:
        raise ValueError(f"Target must contain two classes after cleaning; found {y.value_counts().to_dict()}")

    # Remove likely identifiers and leakage columns. Review this list for your dataset.
    id_like = [c for c in df.columns if c.lower() in {
        "flightid", "id", "tail_number", "tailnumber", "flight_number"
    }]
    df = df.drop(columns=id_like, errors="ignore")

    # Convert obvious numeric-looking object columns when most values parse.
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.90:
            df[col] = converted

    # Keep columns that have at least one non-missing value.
    df = df.dropna(axis=1, how="all")
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced_subsample",
            n_jobs=-1, min_samples_leaf=2
        )
    }

    results = []
    fitted = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])
        print(f"\nTraining {name}...")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        scores = pipe.predict_proba(X_test)[:, 1]
        result = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, scores)
        }
        results.append(result)
        fitted[name] = pipe
        print("\n", name, result)
        print(classification_report(y_test, pred, zero_division=0))

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
        ax.set_title(f"{name}: Confusion Matrix")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{name}_confusion_matrix.png", dpi=150)
        plt.close(fig)

    # ROC curves for both models.
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
    ax.set_title("ROC Curves")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    best_name = results_df.iloc[0]["model"]
    joblib.dump(fitted[best_name], MODEL_DIR / "flight_delay_model.pkl")
    print("\nModel comparison saved to:", OUTPUT_DIR / "model_comparison.csv")
    print("Best model by ROC-AUC:", best_name)
    print("Saved model:", MODEL_DIR / "flight_delay_model.pkl")
    print("Saved evaluation images in:", OUTPUT_DIR)
    print("\nUse the actual values in model_comparison.csv in your report; do not guess metrics.")

if __name__ == "__main__":
    main()
