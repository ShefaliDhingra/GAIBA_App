# ============================
# app.py
# ============================
import streamlit as st
from backend import load_enriched_df, train_numeric_model, predict_candidate_score

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="AIRRA", layout="wide")
st.title("AI Resume vs Job Description Matcher")

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
# Prediction
# ----------------------------
if st.button("Evaluate"):
    if not resume_text or not jd_text or not job_role:
        st.warning("Please enter Resume, Job Description, and Job Role.")
    else:
        output = predict_candidate_score(
            model, vectorizer, role_encoder,
            resume_text, jd_text, job_role, projects_text
        )

        # ------------------------
        # Current Resume Strength (Keyword-based score)
        # ------------------------
        st.subheader("Resume Strength")
        keyword_count = len(output['match_keywords'])

        if keyword_count == 0:
            resume_strength = " No Match!"
            color = "red"
            progress_val = 0
        elif 1 <= keyword_count <= 3:
            resume_strength = " Low"
            color = "gold"
            progress_val = 25
        elif 4 <= keyword_count <= 6:
            resume_strength = " Medium"
            color = "orange"
            progress_val = 60
        else:
            resume_strength = " High"
            color = "green"
            progress_val = 100

        # Display colored heading
        st.markdown(
            f"<h4 style='color:{color}'>{resume_strength}</h4>",
            unsafe_allow_html=True
        )

        # Visual progress bar
        st.progress(progress_val)

        # ------------------------
        # Display keywords nicely
        # ------------------------
        st.subheader("Match Keywords")
        if output['match_keywords']:
            st.markdown(" ".join([f" `{kw}`" for kw in output['match_keywords']]))
        else:
            st.write("No significant match keywords found.")

        st.subheader("Skill / Keyword Gaps")
        if output['skill_gaps']:
            st.markdown(" ".join([f"⚠️ `{kw}`" for kw in output['skill_gaps']]))
        else:
            st.write("No significant skill gaps found.")

        # ------------------------
        # Click-to-expand detailed scores
        # ------------------------
        with st.expander("📊 Detailed Scores"):
            scores = [
                'experience_match', 'skills_match', 'project_relevance',
                'tech_match', 'industry_relevance', 'ats_score', 'relevancy'
            ]
            cols = st.columns(4)
            for idx, score_name in enumerate(scores):
                col = cols[idx % 4]
                col.metric(
                    label=score_name.replace("_", " ").title(),
                    value=f"{output[score_name]:.1f}/10"
                )
