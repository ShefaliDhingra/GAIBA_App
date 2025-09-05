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
# Prediction
# ----------------------------
if st.button("Match"):
    # Your updated Low/Medium/High + meter code here

