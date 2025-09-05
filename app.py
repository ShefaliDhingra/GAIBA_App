# ============================
# app.py
# ============================
import streamlit as st
from backend import load_enriched_df, train_numeric_model, predict_candidate_score

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="AIRRA", layout="wide")

# ----------------------------
# Custom Header
# ----------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>
        🤖 AIRRA
    </h1>
    <h3 style='text-align: center; color: gray; margin-top: -10px;'>
       Artificially Intelligent Resume Refinement Assistant
    </h3>
    <hr style="border:1px solid #f0f0f0">
    """,
    unsafe_allow_html=True
)

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
st.subheader("Please Enter your Details Below")
resume_text = st.text_area("Paste Resume Text here:")
jd_text = st.text_area("Paste Job Description here:")
job_role = st.text_input("Job Role:")
projects_text = st.text_area("Optional: Projects / Notable Work:")

st.markdown("<hr style='border:1px solid #f0f0f0'>", unsafe_allow_html=True)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Evaluate"):
    if not resume_text or not jd_text or not job_role:
        st.warning("⚠️ Please enter Resume, Job Description, and Job Role.")
    else:
        output = predict_candidate_score(
            model, vectorizer, role_encoder,
            resume_text, jd_text, job_role, projects_text
        )

        # ------------------------
        # Current Resume Strength (Keyword-based score)
        # ------------------------
        st.subheader("Current Resume Strength")
        keyword_count = len(output['match_keywords'])

        if keyword_count == 0:
            resume_strength = "No Match!"
            color = "red"
            progress_val = 0
        elif 1 <= keyword_count <= 3:
            resume_strength = "Low"
            color = "gold"
            progress_val = 25
        elif 4 <= keyword_count <= 6:
            resume_strength = "Medium"
            color = "orange"
            progress_val = 60
        else:
            resume_strength = "High"
            color = "green"
            progress_val = 100

        # Left-aligned heading
        st.markdown(
            f"<h3 style='color:{color}; text-align:left'>{resume_strength}</h3>",
            unsafe_allow_html=True
        )

        # Custom colored progress bar
        st.markdown(
            f"""
            <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px; width: 100%;">
                <div style="background-color: {color}; height: 100%; width: {progress_val}%; 
                            border-radius: 10px; text-align: right; padding-right: 5px; color: white; 
                            font-weight: bold; font-size: 12px;">
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<hr style='border:1px solid #f0f0f0'>", unsafe_allow_html=True)

        # ------------------------
        # Display keywords nicely (theme-neutral chips)
        # ------------------------
        st.subheader("Match Keywords")
        if output['match_keywords']:
            st.markdown(
                "<div style='line-height:2'>"
                + " ".join([f"<span style='border:1px solid #888; padding:4px 8px; border-radius:8px;'>`{kw}`</span>" 
                            for kw in output['match_keywords']])
                + "</div>",
                unsafe_allow_html=True
            )
        else:
            st.write("No significant match keywords found.")

        # ------------------------
        # Skill Gaps (theme-neutral chips)
        # ------------------------
        st.subheader("Skill / Keyword Gaps")
        if output['skill_gaps']:
            st.markdown(
                "<div style='line-height:2'>"
                + " ".join([f"<span style='border:1px solid #888; padding:4px 8px; border-radius:8px;'>`{kw}`</span>" 
                            for kw in output['skill_gaps']])
                + "</div>",
                unsafe_allow_html=True
            )
        else:
            st.write("No significant skill gaps found.")

        st.markdown("<hr style='border:1px solid #f0f0f0'>", unsafe_allow_html=True)

        # ------------------------
        # Click-to-expand detailed scores
        # ------------------------
        with st.expander("Bifurcated Scores"):
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
