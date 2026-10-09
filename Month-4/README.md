# Month 4 — Content-Based Movie Recommendation System

## Objective
Build a basic movie recommender using movie overview, genres, keywords, and cast, vectorized with TF-IDF and compared using cosine similarity.

## Dataset
Download the **TMDB 5000 Movie Dataset** from Kaggle. This project expects these two CSV files:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Place both files inside the `Month-4` folder beside `movie_recommender.py`. Dataset files are not included in this ZIP.

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

## Build and test
```bash
python movie_recommender.py
```
This creates the `artifacts` directory containing the saved TF-IDF vectorizer, sparse cosine-similarity matrix, and movie tags.

Then run the optional app:
```bash
streamlit run app.py
```

## Deliverables
- Python recommender logic
- Saved vectorizer and similarity matrix (`artifacts/`)
- Optional Streamlit interface
- Screenshot of the app if built
- Brief explanation of design, testing, and limitations

## How it works
1. Merge movies and credits data on movie ID.
2. Parse genres, keywords, and cast names.
3. Combine overview and selected metadata into a single `tags` text field.
4. Clean the text, convert it to TF-IDF vectors, and calculate cosine similarity.
5. Return the five most similar titles for a given movie.

## Notes
- Similarity is based on metadata, not user ratings or watch history.
- Some titles may have incomplete metadata.
- Do not commit large downloaded datasets or generated artifacts unless your mentor asks for them.
