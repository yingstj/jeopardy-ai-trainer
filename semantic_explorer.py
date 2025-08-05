# streamlit_app/semantic_explorer.py

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/all_jeopardy_clues.csv")
        return df.dropna(subset=["clue"])
    except FileNotFoundError:
        st.error("CSV file not found. Make sure all_jeopardy_clues.csv is in the /data folder.")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def compute_embeddings(df, model):
    return model.encode(df["clue"].tolist(), show_progress_bar=False)

st.title("🔍 Semantic Clue Explorer")

query = st.text_input("Enter a topic, keyword, or concept:")

df = load_data()
model = load_model()

if not df.empty and query:
    with st.spinner("Searching for related clues..."):
        embeddings = compute_embeddings(df, model)
        query_embedding = model.encode([query])
        sims = cosine_similarity(query_embedding, embeddings)[0]
        df["similarity"] = sims
        top_matches = df.sort_values("similarity", ascending=False).head(10)

        st.subheader("Top Matching Clues")
        for _, row in top_matches.iterrows():
            st.markdown(f"**Category:** {row['category']}  
"
                        f"**Clue:** {row['clue']}  
"
                        f"**Answer:** *{row['correct_response']}*  
"
                        f"**Round:** {row.get('round', 'Unknown')}  
"
                        f"**Similarity:** {row['similarity']:.2f}")
            st.markdown("---")
