import streamlit as st
from movie_recommender import recommend, MODEL_DIR
import pandas as pd

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬")
st.title("🎬 Content-Based Movie Recommendation System")
st.write("Enter a movie title to find movies with similar descriptions, genres, keywords, and cast.")

if not (MODEL_DIR / "movie_tags.csv").exists():
    st.warning("Model artifacts are missing. First run: python movie_recommender.py")
else:
    title = st.text_input("Movie title", placeholder="e.g., The Dark Knight")
    if st.button("Recommend"):
        if title.strip():
            results = recommend(title.strip(), top_n=5)
            if isinstance(results, str):
                st.info(results)
            else:
                st.subheader("Top recommendations")
                for i, (name, score) in enumerate(results, 1):
                    st.write(f"{i}. **{name}** — similarity score: {score:.3f}")
