# ============================
# backend.py
# ============================

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Utility functions
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

STOPWORDS = set([
    "a","an","the","and","or","in","on","of","with","to","for","is",
    "are","as","by","from","at","be","this","that","will","you","your"
])

def extract_keywords(text, top_n=20):
    words = clean_text(text).split()
    words = [w for w in words if w not in STOPWORDS and not w.isnumeric() and len(w) > 2]
    freq = pd.Series(words).value_counts()
    return list(freq.index[:top_n])

def map_experience_to_numeric(experience_str):
    mapping = {"entry-level":1,"mid-level":5,"senior-level":8,"expert":10}
    return mapping.get(str(experience_str).lower(),0)

def map_education_to_numeric(education_str):
    mapping = {"bachelors degree":1,"masters degree":2,"phd":3}
    return mapping.get(str(education_str).lower(),0)

# ----------------------------
# Load enriched_df
# ----------------------------
def load_enriched_df(file_path="enriched_df.csv"):
    df = pd.read_csv(file_path)
    df['years_experience_numeric'] = df['years_experience'].apply(map_experience_to_numeric)
    df['education_numeric'] = df['education_level'].apply(map_education_to_numeric)
    df['num_skills'] = df['skills'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    df['num_technologies'] = df['technologies'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    df['num_certifications'] = df['certifications'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    
    role_encoder = LabelEncoder()
    df['job_role_encoded'] = role_encoder.fit_transform(df['Job Roles'].fillna('Unknown'))
    
    return df, role_encoder

# ----------------------------
# Feature engineering
# ----------------------------
def prepare_features(df):
    combined_text = df['Resume'].fillna('') + " " + df['Job Description'].fillna('')
    if 'notable_projects' in df.columns:
        combined_text += " " + df['notable_projects'].fillna('').apply(lambda x: " ".join(eval(x)) if x.strip() != '' else '')
    df['combined_text'] = combined_text
    
    vectorizer = TfidfVectorizer(max_features=1000)
    X_text = vectorizer.fit_transform(df['combined_text'])
    
    num_features = np.vstack([
        df['years_experience_numeric'],
        df['education_numeric'],
        df['num_skills'],
        df['num_technologies'],
        df['num_certifications'],
        df['job_role_encoded']
    ]).T
    
    X = hstack([X_text, num_features])
    return X, vectorizer

# ----------------------------
# Train model
# ----------------------------
def train_numeric_model(df):
    X, vectorizer = prepare_features(df)
    target_cols = ['experience_match','skills_match','project_relevance','tech_match','industry_relevance','ats_score','relevancy']
    y = df[target_cols]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, vectorizer

# ----------------------------
# Match & Scope logic
# ----------------------------
def compute_match_scope(resume_text, jd_text):
    resume_keywords = set(extract_keywords(resume_text, top_n=50))
    jd_keywords = set(extract_keywords(jd_text, top_n=50))
    match = list(resume_keywords & jd_keywords)
    scope = list(jd_keywords - resume_keywords)
    return match, scope

# ----------------------------
# Inference
# ----------------------------
def predict_candidate_score(model, vectorizer, role_encoder, resume_text, jd_text, job_role, projects_text=""):
    try:
        job_role_encoded = role_encoder.transform([job_role])[0]
    except:
        job_role_encoded = 0

    # Combined text for TF-IDF
    combined_text = clean_text(resume_text) + " " + clean_text(jd_text) + " " + clean_text(projects_text)
    X_text = vectorizer.transform([combined_text])
    
    # Cosine similarity between resume and JD
    resume_vec = vectorizer.transform([clean_text(resume_text)])
    jd_vec = vectorizer.transform([clean_text(jd_text)])
    cosine_sim = cosine_similarity(resume_vec, jd_vec)[0][0]
    
    # Numeric features: counts + job role + cosine similarity
    num_features = np.array([[0,0,0,0,0,job_role_encoded]]) * 0.2
    num_features = csr_matrix(num_features)  # convert to sparse
    cosine_sim_sparse = csr_matrix([[cosine_sim]])
    num_features = hstack([num_features, cosine_sim_sparse])
    
    # Combine features
    X_input = hstack([X_text, num_features])
    
    numeric_cols = ['experience_match','skills_match','project_relevance','tech_match','industry_relevance','ats_score','relevancy']
    numeric_preds = model.predict(X_input)[0]
    
    match_keywords, skill_gaps = compute_match_scope(resume_text, jd_text)
    
    overall_percentage = numeric_preds.sum() / (len(numeric_cols)*10) * 100
    
    output = {col: float(val) for col,val in zip(numeric_cols, numeric_preds)}
    output.update({
        "overall_percentage": overall_percentage,
        "match_keywords": match_keywords,
        "skill_gaps": skill_gaps
    })
    
    return output
