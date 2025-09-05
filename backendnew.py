import os
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------- Load Models ----------
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

numeric_model_path = os.path.join(MODELS_DIR, "numeric_model.pkl")
vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
encoder_path = os.path.join(MODELS_DIR, "job_role_encoder.pkl")

with open(numeric_model_path, "rb") as f:
    numeric_model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    tfidf_vectorizer: TfidfVectorizer = pickle.load(f)

with open(encoder_path, "rb") as f:
    job_role_encoder = pickle.load(f)

# ---------- Helpers ----------
def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())

def extract_keywords(text, top_n=20):
    """Extract top N keywords from text."""
    words = clean_text(text).split()
    freq = pd.Series(words).value_counts()
    return list(freq.index[:top_n])

# ---------- Inference ----------
def predict_resume_fit(resume_text: str, job_role: str, job_description: str):
    """
    Given a candidate resume, job role, and JD:
    - Vectorizes text
    - Runs numeric model
    - Returns scores + matched/missing keywords
    """

    # Combine resume + JD for vectorization
    combined_text = clean_text(resume_text + " " + job_description)
    vec = tfidf_vectorizer.transform([combined_text])

    # Encode job role
    try:
        job_role_encoded = job_role_encoder.transform([job_role])[0]
    except ValueError:
        job_role_encoded = -1  # unseen role

    # Run prediction
    preds = numeric_model.predict(vec.toarray())
    preds = preds.flatten().tolist() if hasattr(preds, "flatten") else preds.tolist()

    # Extract keywords
    resume_keywords = extract_keywords(resume_text, top_n=15)
    jd_keywords = extract_keywords(job_description, top_n=15)

    matched_keywords = list(set(resume_keywords) & set(jd_keywords))
    missing_keywords = list(set(jd_keywords) - set(resume_keywords))

    return {
        "experience_match": preds[0] if len(preds) > 0 else None,
        "skills_match": preds[1] if len(preds) > 1 else None,
        "project_relevance": preds[2] if len(preds) > 2 else None,
        "tech_match": preds[3] if len(preds) > 3 else None,
        "industry_relevance": preds[4] if len(preds) > 4 else None,
        "ats_score": preds[5] if len(preds) > 5 else None,
        "relevancy": preds[6] if len(preds) > 6 else None,
        "match_keywords": matched_keywords,
        "skill_gaps": missing_keywords,
    }
