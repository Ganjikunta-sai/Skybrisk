from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA = Path(__file__).parent / "netflix_titles.csv"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
print("Shape:", df.shape)
df.info()
print(df.describe(include="all").T)
print("Missing values:\n", df.isna().sum().sort_values(ascending=False).head(20))
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

for c in ["director","cast","country","rating","listed_in"]:
    if c in df:
        df[c] = df[c].fillna("Unknown")
if "date_added" in df:
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
if "release_year" in df:
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df["release_decade"] = ((df["release_year"] // 10) * 10).astype("Int64").astype(str) + "s"
if "type" in df:
    df["content_type"] = df["type"].astype(str).str.strip()
    df["is_movie"] = (df["content_type"].str.lower() == "movie").astype(int)

df.to_csv(OUT/"netflix_titles_cleaned.csv", index=False)
sns.set_theme(style="whitegrid")

if "content_type" in df:
    plt.figure(figsize=(6,6)); df["content_type"].value_counts().plot.pie(autopct="%1.1f%%")
    plt.ylabel(""); plt.title("Movies vs TV Shows"); plt.tight_layout()
    plt.savefig(OUT/"01_content_type_pie.png", dpi=150); plt.close()
if "release_year" in df:
    yearly = df.dropna(subset=["release_year"]).groupby("release_year").size().sort_index()
    plt.figure(figsize=(10,4)); yearly.plot()
    plt.title("Titles by Release Year"); plt.xlabel("Release year"); plt.ylabel("Number of titles")
    plt.tight_layout(); plt.savefig(OUT/"02_titles_by_release_year.png", dpi=150); plt.close()
if "listed_in" in df:
    genres = df["listed_in"].dropna().str.split(",").explode().str.strip()
    genres = genres[genres.ne("Unknown")]
    top = genres.value_counts().head(10).sort_values()
    plt.figure(figsize=(9,5)); top.plot.barh(); plt.title("Top 10 Genres/Categories")
    plt.xlabel("Title-category appearances"); plt.tight_layout()
    plt.savefig(OUT/"03_top_genres.png", dpi=150); plt.close()
if "country" in df:
    countries = df["country"].dropna().str.split(",").explode().str.strip()
    countries = countries[countries.ne("Unknown")]
    top = countries.value_counts().head(10).sort_values()
    plt.figure(figsize=(9,5)); top.plot.barh(); plt.title("Top 10 Countries")
    plt.xlabel("Title-country appearances"); plt.tight_layout()
    plt.savefig(OUT/"04_top_countries.png", dpi=150); plt.close()
if {"release_decade","content_type"}.issubset(df.columns):
    table = pd.crosstab(df["release_decade"], df["content_type"])
    plt.figure(figsize=(9,6)); sns.heatmap(table, cmap="Blues")
    plt.title("Release Decade vs Content Type"); plt.tight_layout()
    plt.savefig(OUT/"05_decade_type_heatmap.png", dpi=150); plt.close()
print("Cleaned CSV and plots saved in:", OUT)
print("Write 5–6 insights based on actual charts; metadata is not viewing data.")
