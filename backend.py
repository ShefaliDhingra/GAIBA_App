import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle


# -------------------------
# Utility functions
# -------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_match_score(vectorizer, text1, text2):
    texts = [clean_text(text1), clean_text(text2)]
    tfidf = vectorizer.transform(texts)
    return cosine_similarity(tfidf[0], tfidf[1])[0][0]


def load_enriched_df(file_path):
    return pd.read_csv(file_path)


# -------------------------
# New predict_candidate_score
# -------------------------
def predict_candidate_score(model, vectorizer, role_encoder,
                            resume_text, jd_text, job_role, projects_text="",
                            years_experience=None, education_level=None):
    """
    Predicts a candidate–job match score using resume, JD, job role, and extras.
    Uses a relative scoring baseline (compares to random role instead of raw cosine sim).
    """

    # Clean inputs
    resume_text_clean = clean_text(resume_text)
    jd_text_clean = clean_text(jd_text)
    projects_text_clean = clean_text(projects_text)

    # Compute cosine similarity between resume and JD
    sim_score = compute_match_score(vectorizer, resume_text_clean, jd_text_clean)

    # Add project weight if present
    if projects_text_clean.strip():
        proj_score = compute_match_score(vectorizer, resume_text_clean, projects_text_clean)
        sim_score = (sim_score * 0.8) + (proj_score * 0.2)

    # 🔑 Relative baseline: compare against random role
    roles = list(role_encoder.classes_)
    if job_role in roles:
        fake_role = np.random.choice([r for r in roles if r != job_role])  # pick a different role
    else:
        fake_role = np.random.choice(roles)

    fake_role_text = clean_text(fake_role)
    baseline_score = compute_match_score(vectorizer, resume_text_clean, fake_role_text)

    # Relative score = candidate vs JD minus baseline
    relative_score = sim_score - baseline_score

    # Clip between 0 and 1 for stability
    relative_score = max(0.0, min(1.0, relative_score))

    # Prepare feature vector for ML model
    try:
        job_role_encoded = role_encoder.transform([job_role])[0]
    except Exception:
        job_role_encoded = -1

    features = [relative_score, job_role_encoded]

    if years_experience is not None:
        features.append(years_experience)
    if education_level is not None:
        features.append(education_level)

    features = np.array(features).reshape(1, -1)

    # Predict with ML model
    try:
        ml_score = model.predict_proba(features)[0][1]  # probability of match
    except Exception:
        ml_score = relative_score

    # Final score = weighted combo
    final_score = (0.6 * ml_score) + (0.4 * relative_score)

    output = {
        "similarity_score": float(sim_score),
        "baseline_score": float(baseline_score),
        "relative_score": float(relative_score),
        "ml_score": float(ml_score),
        "final_score": float(final_score * 100)  # percentage
    }

    return output


# -------------------------
# Compatibility stub
# -------------------------
def train_numeric_model(df):
    """
    Placeholder for compatibility with app.py.
    Not used in current workflow.
    """
    return None

