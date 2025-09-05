# ============================
# backend.py
# ============================

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack

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
# Weighted Inference
# ----------------------------
def predict_candidate_score(model, vectorizer, role_encoder, resume_text, jd_text, job_role, years_experience=None, education_level=None, projects_text=""):
    # Encode job role
    try:
        job_role_encoded = role_encoder.transform([job_role])[0]
    except:
        job_role_encoded = 0
    
    # Numeric fields
    exp_numeric = map_experience_to_numeric(years_experience) if years_experience else 0
    edu_numeric = map_education_to_numeric(education_level) if education_level else 0
    num_skills = 0
    num_technologies = 0
    num_certifications = 0
    
    # TF-IDF text
    combined_text = clean_text(resume_text) + " " + clean_text(jd_text) + " " + clean_text(projects_text)
    X_text = vectorizer.transform([combined_text])
    
    # Numeric features
    num_features = np.array([[exp_numeric, edu_numeric, num_skills, num_technologies, num_certifications, job_role_encoded]])
    from scipy.sparse import csr_matrix
    num_features = csr_matrix(num_features)
    
    # Combine features
    X_input = hstack([X_text, num_features])
    
    numeric_cols = ['experience_match','skills_match','project_relevance','tech_match','industry_relevance','ats_score','relevancy']
    
    # Predict
    try:
        numeric_preds = model.predict(X_input)[0]
    except:
        numeric_preds = np.zeros(len(numeric_cols))
    
    # Keyword matching
    match_keywords, skill_gaps = compute_match_scope(resume_text, jd_text) if resume_text and jd_text else ([], [])
    
    # ----------------------------
    # Weighted Overall Percentage
    # ----------------------------
    # Define your weights here (sum can be <=1)
    weights = {
        'experience_match': 0.2,
        'skills_match': 0.25,
        'project_relevance': 0.15,
        'tech_match': 0.1,
        'industry_relevance': 0.15,
        'ats_score': 0.1,
        'relevancy': 0.05
    }
    
    overall_percentage = sum(numeric_preds[i] * weights[col] for i, col in enumerate(numeric_cols)) / 10 * 100
    
    output = {col: float(val) for col,val in zip(numeric_cols, numeric_preds)}
    output.update({
        "overall_percentage": overall_percentage,
        "match_keywords": match_keywords,
        "skill_gaps": skill_gaps
    })
    
    return output
