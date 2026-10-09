from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA = Path(__file__).parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
print("Shape:", df.shape)
df.info()
print(df.describe(include="all").T)
print("Missing values:\n", df.isna().sum().sort_values(ascending=False).head(20))

df = df.replace(r"^\\s*$", pd.NA, regex=True)
if "TotalCharges" in df:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
if "Churn" in df:
    df = df.dropna(subset=["Churn"])

for c in df.select_dtypes(include="number"):
    df[c] = df[c].fillna(df[c].median())
for c in df.select_dtypes(include=["object", "category"]):
    mode = df[c].mode(dropna=True)
    if not mode.empty:
        df[c] = df[c].fillna(mode.iloc[0])

if "tenure" in df:
    df["TenureGroup"] = pd.cut(df["tenure"], [-1,12,24,48,60,float("inf")],
        labels=["0-12 months","13-24 months","25-48 months","49-60 months","61+ months"])
if {"TotalCharges","tenure"}.issubset(df.columns):
    df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].replace(0, float("nan"))
    if "MonthlyCharges" in df:
        df["AvgMonthlySpend"] = df["AvgMonthlySpend"].fillna(df["MonthlyCharges"])

for c in df.select_dtypes(include="object"):
    vals = set(df[c].dropna().astype(str).str.lower().unique())
    if vals and vals.issubset({"yes","no"}):
        df[c] = df[c].astype(str).str.lower().map({"yes":1,"no":0})

df.to_csv(OUT / "customer_churn_cleaned.csv", index=False)
sns.set_theme(style="whitegrid")

if "Churn" in df:
    plt.figure(figsize=(6,4)); sns.countplot(data=df, x="Churn")
    plt.title("Customer Churn Distribution"); plt.tight_layout()
    plt.savefig(OUT/"01_churn_countplot.png", dpi=150); plt.close()
if {"Contract","Churn"}.issubset(df.columns):
    plt.figure(figsize=(8,4)); sns.countplot(data=df, x="Contract", hue="Churn")
    plt.title("Contract vs Churn"); plt.xticks(rotation=15); plt.tight_layout()
    plt.savefig(OUT/"02_contract_vs_churn.png", dpi=150); plt.close()
if {"MonthlyCharges","Churn"}.issubset(df.columns):
    plt.figure(figsize=(6,4)); sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
    plt.title("Monthly Charges by Churn"); plt.tight_layout()
    plt.savefig(OUT/"03_monthly_charges_by_churn.png", dpi=150); plt.close()
corr = df.select_dtypes(include="number").corr(numeric_only=True)
if not corr.empty:
    plt.figure(figsize=(10,7)); sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Numeric Correlation Heatmap"); plt.tight_layout()
    plt.savefig(OUT/"04_correlation_heatmap.png", dpi=150); plt.close()
print("Cleaned CSV and plots saved in:", OUT)
print("Write five insights based on your actual charts and counts.")
