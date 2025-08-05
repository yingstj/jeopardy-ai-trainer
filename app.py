# streamlit_app/app.py

import streamlit as st
import pandas as pd
import random
import re
import os
import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Normalize text for fuzzy matching
def normalize(text):
    text = text.lower()
    text = re.sub(r"^(what|who|where|when|why|how)\s+(is|are|was|were)\s+", "", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text.strip()

# Load and filter data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/all_jeopardy_clues.csv")
        df = df.dropna(subset=["clue", "correct_response"])
        df["clue_embedding"] = df["clue"].apply(lambda x: model.encode(x))
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# Load or initialize user progress
progress_file = "data/user_progress.csv"
if os.path.exists(progress_file):
    progress_df = pd.read_csv(progress_file)
else:
    progress_df = pd.DataFrame(columns=["date", "total", "correct"])

# Track session state
if "history" not in st.session_state:
    st.session_state.history = []

if "score" not in st.session_state:
    st.session_state.score = 0
    st.session_state.total = 0

if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.datetime.now()

if "current_clue" not in st.session_state:
    st.session_state.current_clue = None

st.title("🧠 Jay's Jeopardy Trainer")

df = load_data()

if df.empty:
    st.warning("⚠️ CSV file not found. Make sure all_jeopardy_clues.csv is in the /data folder.")
    st.info("Once your dataset finishes downloading, relaunch this app!")
    st.stop()

categories = sorted(df["category"].unique())
selected_categories = st.multiselect("Filter by category:", categories, default=categories[:3])
filtered_df = df[df["category"].isin(selected_categories)]

if st.session_state.current_clue is None:
    st.session_state.current_clue = random.choice(filtered_df.to_dict(orient="records"))
    st.session_state.start_time = datetime.datetime.now()

clue = st.session_state.current_clue
st.subheader(f"📚 Category: {clue['category']}")
st.markdown(f"**Clue:** {clue['clue']}")

time_limit = st.slider("⏱️ Time Limit (seconds):", 10, 60, 30)

with st.form(key="clue_form"):
    user_input = st.text_input("Your response:")
    submitted = st.form_submit_button("Submit")

if submitted:
    elapsed_time = (datetime.datetime.now() - st.session_state.start_time).seconds
    user_clean = normalize(user_input)
    answer_clean = normalize(clue["correct_response"])
    correct = user_clean == answer_clean and elapsed_time <= time_limit

    if correct:
        st.success("✅ Correct!")
        st.session_state.score += 1
    else:
        st.error(f"❌ Incorrect or timed out. The correct response was: *{clue['correct_response']}*")

        # Semantic similarity
        user_vector = model.encode(clue["clue"])
        clue_vectors = np.vstack(filtered_df["clue_embedding"].values)
        similarities = cosine_similarity([user_vector], clue_vectors)[0]
        top_indices = similarities.argsort()[-4:][::-1]
        similar_clues = filtered_df.iloc[top_indices]

        with st.expander("🔍 Review similar clues"):
            for _, row in similar_clues.iterrows():
                st.markdown(f"- **{row['category']}**: {row['clue']}")
                st.markdown(f"  → *{row['correct_response']}*")

    st.session_state.total += 1
    st.session_state.history.append({
        "game_id": clue.get("game_id", ""),
        "category": clue["category"],
        "clue": clue["clue"],
        "correct_response": clue["correct_response"],
        "round": clue.get("round", ""),
        "user_response": user_input,
        "was_correct": correct
    })

    today = datetime.date.today().isoformat()
    new_row = pd.DataFrame([[today, 1, 1 if correct else 0]], columns=["date", "total", "correct"])
    progress_df = pd.concat([progress_df, new_row], ignore_index=True)
    progress_df.to_csv(progress_file, index=False)

    st.session_state.current_clue = None
    st.experimental_rerun()

if st.session_state.total:
    st.markdown("---")
    st.metric("Your Score", f"{st.session_state.score} / {st.session_state.total}")

if st.session_state.history:
    st.subheader("📊 Session Recap")
    st.dataframe(pd.DataFrame(st.session_state.history))

    with st.expander("📈 Progress Tracker"):
        summary = progress_df.groupby("date").sum().reset_index()
        summary["accuracy"] = (summary["correct"] / summary["total"]).round(2)
        st.dataframe(summary)

    st.markdown("---")
    if st.button("🔁 Adaptive Retry Mode"):
        missed = [h for h in st.session_state.history if not h["was_correct"]]
        if missed:
            retry = random.choice(missed)
            st.session_state.current_clue = retry
            st.experimental_rerun()
        else:
            st.info("No missed clues yet to retry!")