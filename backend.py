# ============================
# backend.py
# ============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# ----------------------------
# Utility functions
# ----------------------------
def map_experience_to_numeric(experience_str):
    mapping = {"entry-level":1,"mid-level":4,"senior-level":6,"expert":8}
    return mapping.get(str(experience_str).lower(),0)

def map_education_to_numeric(education_str):
    mapping = {"bachelors degree":1,"masters degree":2,"phd":3}
    return mapping.get(str(education_str).lower(),0)

# ----------------------------
# Load enriched_df
# ----------------------------
def load_enriched_df(file_path="enriched_df.csv"):
    df = pd.read_csv(file_path)
    
    # Numeric feature preparation
    df['years_experience_numeric'] = df['years_experience'].apply(map_experience_to_numeric)
    df['education_numeric'] = df['education_level'].apply(map_education_to_numeric)
    df['num_skills'] = df['skills'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    df['num_technologies'] = df['technologies'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    df['num_certifications'] = df['certifications'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    
    # Encode job roles
    role_encoder = LabelEncoder()
    df['job_role_encoded'] = role_encoder.fit_transform(df['Job Roles'].fillna('Unknown'))
    
    return df, role_encoder

# ----------------------------
# Train numeric model
# ----------------------------
def train_numeric_model(df):
    features = ['years_experience_numeric', 'education_numeric',
                'num_skills', 'num_technologies', 'num_certifications',
                'job_role_encoded']
    target_cols = ['experience_match','skills_match','project_relevance',
                   'tech_match','industry_relevance','ats_score','relevancy']
    
    X = df[features]
    y = df[target_cols]
    
    # Train LightGBM Regressor
    model = lgb.LGBMRegressor(objective='regression', n_estimators=500, learning_rate=0.05)
    model.fit(X, y)
    
    return model, features

# ----------------------------
# Optional: Keyword Matching (minimal)
# ----------------------------
import re

STOPWORDS = set([
    "a","an","the","and","or","in","on","of","with","to","for","is",
    "are","as","by","from","at","be","this","that","will","you","your"
])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_keywords(text, top_n=20):
    words = clean_text(text).split()
    words = [w for w in words if w not in STOPWORDS and not w.isnumeric() and len(w) > 2]
    freq = pd.Series(words).value_counts()
    return list(freq.index[:top_n])

def compute_match_scope(resume_text, jd_text):
    resume_keywords = set(extract_keywords(resume_text, top_n=50))
    jd_keywords = set(extract_keywords(jd_text, top_n=50))
    match = list(resume_keywords & jd_keywords)
    scope = list(jd_keywords - resume_keywords)
    return match, scope

# ----------------------------
# Predict candidate score
# ----------------------------
def predict_candidate_score(model, features, role_encoder,
                            years_experience=0,
                            education_level=0,
                            num_skills=0,
                            num_technologies=0,
                            num_certifications=0,
                            job_role="Unknown",
                            resume_text="",
                            jd_text=""):
    
    try:
        job_role_encoded = role_encoder.transform([job_role])[0]
    except:
        job_role_encoded = 0
    
    # Build input DataFrame
    X_input = pd.DataFrame([[years_experience, education_level,
                             num_skills, num_technologies, num_certifications,
                             job_role_encoded]], columns=features)
    
    # Predict numeric scores
    numeric_cols = ['experience_match','skills_match','project_relevance',
                    'tech_match','industry_relevance','ats_score','relevancy']
    
    numeric_preds = model.predict(X_input)[0]
    
    # Compute overall percentage
    overall_percentage = numeric_preds.sum() / (len(numeric_cols)*10) * 100
    
    # Optional: compute keyword match (if resume and JD provided)
    if resume_text and jd_text:
        match_keywords, skill_gaps = compute_match_scope(resume_text, jd_text)
    else:
        match_keywords, skill_gaps = [], []
    
    output = {col: float(val) for col,val in zip(numeric_cols, numeric_preds)}
    output.update({
        "overall_percentage": overall_percentage,
        "match_keywords": match_keywords,
        "skill_gaps": skill_gaps
    })
    
    return output
