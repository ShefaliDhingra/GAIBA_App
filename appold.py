# ============================
# app.py
# ============================
import streamlit as st
import pandas as pd
from backend import load_enriched_df, train_numeric_model, predict_candidate_score

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="AI Resume Matcher", layout="wide")
st.title("AI Resume vs Job Description Matcher")

# ----------------------------
# Load Data and Train Model
# ----------------------------
@st.cache_data
def load_data_and_train():
    df, role_encoder = load_enriched_df("enriched_df.csv")
    model, vectorizer = train_numeric_model(df)
    return df, model, vectorizer, role_encoder

with st.spinner("Loading data and training model..."):
    df, model, vectorizer, role_encoder = load_data_and_train()

st.success("Model ready!")

# ----------------------------
# Input Section
# ----------------------------
st.header("Candidate Input")
resume_text = st.text_area("Paste Resume Text here:")
jd_text = st.text_area("Paste Job Description here:")
job_role = st.text_input("Job Role:")
projects_text = st.text_area("Optional: Projects / Notable Work:")

# ----------------------------
# Predict Button
# ----------------------------
if st.button("Predict Match"):
    if resume_text.strip() == "" or jd_text.strip() == "" or job_role.strip() == "":
        st.error("Please provide Resume, Job Description, and Job Role.")
    else:
        with st.spinner("Predicting candidate score..."):
            output = predict_candidate_score(resume_text, jd_text, job_role, projects_text)

        st.subheader("✅ Match Scores")
        score_cols = ['experience_match', 'skills_match', 'project_relevance',
                      'tech_match', 'industry_relevance', 'ats_score', 'relevancy']
        for col in score_cols:
            st.metric(label=col.replace("_", " ").title(), value=f"{output[col]:.1f}/10")

        st.metric(label="Overall Percentage", value=f"{output['overall_percentage']:.1f}%")

        st.subheader("Match Keywords")
        st.write(", ".join(output['match_keywords'][:20]))

        st.subheader("Skill / Keyword Gaps")
        st.write(", ".join(output['skill_gaps'][:20]))
