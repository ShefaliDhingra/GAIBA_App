# ============================
# app.py
# ============================
import streamlit as st
from backend import load_enriched_df, train_numeric_model, predict_candidate_score

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="AI Resume vs Job Description Matcher", layout="wide")
st.title("AI Resume vs Job Description Matcher")
st.write("Model ready!")

# ----------------------------
# Load Data and Train Model
# ----------------------------
@st.cache_data(show_spinner=True)
def load_and_train():
    df, role_encoder = load_enriched_df("enriched_df.csv")
    model, vectorizer = train_numeric_model(df)
    return model, vectorizer, role_encoder

model, vectorizer, role_encoder = load_and_train()

# ----------------------------
# Candidate Input
# ----------------------------
st.header("Candidate Input")
resume_text = st.text_area("Paste Resume Text here:")
jd_text = st.text_area("Paste Job Description here:")
job_role = st.text_input("Job Role:")
projects_text = st.text_area("Optional: Projects / Notable Work:")

# ----------------------------
# Helper to convert numeric score to Low/Medium/High
# ----------------------------
def convert_score_label(score):
    if score < 8:
        return "Low"
    elif 8 <= score <= 9:
        return "Medium"
    else:
        return "High"

# ----------------------------
# Prediction
# ----------------------------
if st.button("Match"):
    if not resume_text or not jd_text or not job_role:
        st.warning("Please enter Resume, Job Description, and Job Role.")
    else:
        output = predict_candidate_score(model, vectorizer, role_encoder,
                                         resume_text, jd_text, job_role, projects_text)
        
        st.success("✅ Match Scores (Labels)")

        # Display numeric scores as Low/Medium/High
        st.subheader("Experience Match")
        st.write(convert_score_label(output['experience_match']))

        st.subheader("Skills Match")
        st.write(convert_score_label(output['skills_match']))

        st.subheader("Project Relevance")
        st.write(convert_score_label(output['project_relevance']))

        st.subheader("Tech Match")
        st.write(convert_score_label(output['tech_match']))

        st.subheader("Industry Relevance")
        st.write(convert_score_label(output['industry_relevance']))

        st.subheader("ATS Score")
        st.write(convert_score_label(output['ats_score']))

        st.subheader("Relevancy")
        st.write(convert_score_label(output['relevancy']))

        # Detailed analysis in an expander
        with st.expander("Click here for Detailed Analysis"):
            st.subheader("Overall Percentage")
            st.write(f"{output['overall_percentage']:.1f}%")

            st.subheader("Match Keywords")
            st.write(", ".join(output['match_keywords']))

            st.subheader("Skill / Keyword Gaps")
            st.write(", ".join(output['skill_gaps']))
