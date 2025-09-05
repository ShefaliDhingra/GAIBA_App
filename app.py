# ============================
# app.py
# ============================

import streamlit as st
from backend import load_enriched_df, train_numeric_model, predict_candidate_score, map_experience_to_numeric, map_education_to_numeric

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="AI Resume vs Job Description Matcher", layout="wide")
st.title("AI Resume vs Job Description Matcher")

# ----------------------------
# Load Data and Train Model
# ----------------------------
@st.cache_data(show_spinner=True)
def load_and_train():
    df, role_encoder = load_enriched_df("enriched_df.csv")
    model, features = train_numeric_model(df)
    return model, features, role_encoder

model, features, role_encoder = load_and_train()

# ----------------------------
# Candidate Input
# ----------------------------
st.header("Candidate Input")

years_experience = st.number_input("Years of Experience", min_value=0, max_value=30, value=5)
education_level = st.selectbox("Education Level", ["Bachelors Degree","Masters Degree","PhD"])
num_skills = st.number_input("Number of Skills", min_value=0, max_value=50, value=5)
num_technologies = st.number_input("Number of Technologies", min_value=0, max_value=50, value=2)
num_certifications = st.number_input("Number of Certifications", min_value=0, max_value=20, value=1)
job_role = st.text_input("Job Role")

# Map education level to numeric
education_numeric = map_education_to_numeric(education_level)
years_experience_numeric = years_experience  # already numeric

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Match"):
    if not job_role:
        st.warning("Please enter a Job Role.")
    else:
        output = predict_candidate_score(
            model=model,
            features=features,
            role_encoder=role_encoder,
            years_experience=years_experience_numeric,
            education_level=education_numeric,
            num_skills=num_skills,
            num_technologies=num_technologies,
            num_certifications=num_certifications,
            job_role=job_role
        )

        st.success("✅ Match Scores")

        # ------------------------
        # Overall Percentage Meter
        # ------------------------
        overall = output['overall_percentage']
        st.subheader("Overall Match")
        st.progress(int(overall))
        
        # Define bracket labels
        if overall < 60:
            bracket = "Very Low chances"
            color = "red"
        elif overall <= 75:
            bracket = "Low chances"
            color = "orange"
        elif overall <= 85:
            bracket = "Medium chances"
            color = "yellow"
        elif overall <= 95:
            bracket = "High chances"
            color = "lightgreen"
        else:
            bracket = "Very High chances"
            color = "green"
        
        st.markdown(f"<h4 style='color:{color}'>{overall:.1f}% - {bracket}</h4>", unsafe_allow_html=True)

        # ------------------------
        # Display numeric scores in columns
        # ------------------------
        st.subheader("Detailed Scores")
        scores = ['experience_match','skills_match','project_relevance','tech_match','industry_relevance','ats_score','relevancy']
        cols = st.columns(4)
        for idx, score_name in enumerate(scores):
            col = cols[idx % 4]
            col.metric(label=score_name.replace("_"," ").title(), value=f"{output[score_name]:.1f}/10")
