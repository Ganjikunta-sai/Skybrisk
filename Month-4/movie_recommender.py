from pathlib import Path
import ast
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
MOVIES_CSV = ROOT / "tmdb_5000_movies.csv"
CREDITS_CSV = ROOT / "tmdb_5000_credits.csv"
MODEL_DIR = ROOT / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)

def parse_names(value, key="name"):
    """Convert TMDB JSON-like string fields to a list of names."""
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
        return [str(item.get(key, "")) for item in parsed if isinstance(item, dict) and item.get(key)]
    except (ValueError, SyntaxError, TypeError):
        return []

def clean_text(value):
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9\\s]", " ", value)
    return re.sub(r"\\s+", " ", value).strip()

def main():
    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

    # The movies and credits files share the movie title and id.
    credits = credits.rename(columns={"movie_id": "id"})
    df = movies.merge(credits[["id", "cast", "crew"]], on="id", how="left")

    required = {"title", "overview", "genres", "keywords", "cast"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset columns missing: {sorted(missing)}")

    df["overview"] = df["overview"].fillna("")
    for col in ["genres", "keywords", "cast"]:
        df[col + "_names"] = df[col].apply(parse_names)

    # Use top 3 cast names to keep the feature space more useful and manageable.
    df["cast_names"] = df["cast"].apply(lambda x: x[:3])
    df["tags"] = (
        df["overview"].astype(str) + " " +
        df["genres_names"].apply(lambda x: " ".join(x)) + " " +
        df["keywords_names"].apply(lambda x: " ".join(x)) + " " +
        df["cast_names"].apply(lambda x: " ".join(x))
    ).apply(clean_text)

    # Drop records without a title or useful tag text.
    df = df.dropna(subset=["title"]).reset_index(drop=True)
    df = df[df["tags"].str.strip().ne("")].reset_index(drop=True)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    matrix = vectorizer.fit_transform(df["tags"])
    similarity = cosine_similarity(matrix, dense_output=False)

    with open(MODEL_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    # Save a sparse similarity matrix to reduce disk use.
    with open(MODEL_DIR / "cosine_similarity.pkl", "wb") as f:
        pickle.dump(similarity, f)
    df[["id", "title", "tags"]].to_csv(MODEL_DIR / "movie_tags.csv", index=False)

    print(f"Prepared {len(df)} movies.")
    print("Artifacts saved in:", MODEL_DIR)
    print("Try from Python:")
    print("from movie_recommender import recommend")
    print("print(recommend('The Dark Knight'))")

def recommend(title, top_n=5):
    """Return the top N similar movie titles for an exact or partial title match."""
    with open(MODEL_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(MODEL_DIR / "cosine_similarity.pkl", "rb") as f:
        similarity = pickle.load(f)
    df = pd.read_csv(MODEL_DIR / "movie_tags.csv")

    matches = df[df["title"].str.casefold() == title.casefold()]
    if matches.empty:
        matches = df[df["title"].str.contains(re.escape(title), case=False, na=False)]
    if matches.empty:
        return f"No movie found matching: {title}"

    index = int(matches.index[0])
    scores = list(enumerate(similarity.getrow(index).toarray().ravel()))
    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    results = [(df.iloc[i]["title"], float(score)) for i, score in scores if i != index][:top_n]
    return results

if __name__ == "__main__":
    main()
