import streamlit as st
from backend import predict_resume_fit

# ---------------- App Layout ----------------
st.set_page_config(page_title="AI Resume Matcher", layout="wide")

st.title("📄 AI-Powered Resume Matcher")
st.markdown("Upload your resume, paste a Job Description, and see how well you match!")

# ---------------- Inputs ----------------
resume_file = st.file_uploader("Upload Resume (TXT or DOCX)", type=["txt", "docx"])
job_role = st.text_input("Enter Job Role")
job_description = st.text_area("Paste Job Description")

# Extract resume text
resume_text = ""
if resume_file:
    if resume_file.name.endswith(".txt"):
        resume_text = resume_file.read().decode("utf-8")
    elif resume_file.name.endswith(".docx"):
        import docx
        doc = docx.Document(resume_file)
        resume_text = "\n".join([p.text for p in doc.paragraphs])

# ---------------- Prediction ----------------
if st.button("🔍 Evaluate Resume") and resume_text and job_role and job_description:
    with st.spinner("Analyzing... Please wait"):
        results = predict_resume_fit(resume_text, job_role, job_description)

    # ---------------- Results ----------------
    st.subheader("📊 Evaluation Results")

    # Numeric scores
    score_labels = [
        "Experience Match",
        "Skills Match",
        "Project Relevance",
        "Technologies/Tools Match",
        "Industry/Domain Relevance",
        "ATS Score",
        "Relevancy"
    ]

    numeric_scores = [
        results["experience_match"],
        results["skills_match"],
        results["project_relevance"],
        results["tech_match"],
        results["industry_relevance"],
        results["ats_score"],
        results["relevancy"]
    ]

    for label, score in zip(score_labels, numeric_scores):
        if score is not None:
            st.write(f"**{label}:** {score}/10")
            st.progress(min(int(score) / 10, 1.0))

    # Keywords matched
    st.subheader("✅ Matched Keywords")
    st.write(", ".join(results["match_keywords"]) if results["match_keywords"] else "No strong matches found.")

    # Skill gaps
    st.subheader("⚠️ Skill Gaps (Missing Keywords)")
    st.write(", ".join(results["skill_gaps"]) if results["skill_gaps"] else "No major skill gaps identified.")

else:
    st.info("Please upload a resume, enter job role, and paste a job description.")
