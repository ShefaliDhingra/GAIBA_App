# ============================
# backend.py
# ============================
"""
Backend for Resume vs Job Matching AI.

Responsibilities:
1. Load enriched_df.csv (all 150 entries)
2. Feature engineering (text + structured numeric features + Job Role encoding)
3. TF-IDF vectorizer creation
4. Multi-output regression model for numeric targets
5. Saving trained model, vectorizer, and Job Role encoder
6. Inference function for new candidate input (resume + JD + role)
7. Keyword-based logic for Match and Scope
"""

# ----------------------------
# Imports
# ----------------------------
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack

# ----------------------------
# Section 1: Utility functions
# ----------------------------
def clean_text(text):
    """Lowercase, remove punctuation and extra spaces"""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_keywords(text, top_n=20):
    """Extract top N frequent words from a single text"""
    words = clean_text(text).split()
    freq = pd.Series(words).value_counts()
    return list(freq.index[:top_n])

def map_experience_to_numeric(experience_str):
    mapping = {
        "entry-level": 1,
        "mid-level": 5,
        "senior-level": 8,
        "expert": 10
    }
    return mapping.get(str(experience_str).lower(), 0)

def map_education_to_numeric(education_str):
    mapping = {
        "bachelors degree": 1,
        "masters degree": 2,
        "phd": 3
    }
    return mapping.get(str(education_str).lower(), 0)

# ----------------------------
# Section 2: Load enriched_df
# ----------------------------
def load_enriched_df(file_path="enriched_df.csv"):
    df = pd.read_csv(file_path)
    
    # Numeric conversions
    df['years_experience_numeric'] = df['years_experience'].apply(map_experience_to_numeric)
    df['education_numeric'] = df['education_level'].apply(map_education_to_numeric)
    
    # Counts of skills, techs, certifications
    df['num_skills'] = df['skills'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    df['num_technologies'] = df['technologies'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    df['num_certifications'] = df['certifications'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    
    # Encode Job Roles
    role_encoder = LabelEncoder()
    df['job_role_encoded'] = role_encoder.fit_transform(df['Job Roles'].fillna('Unknown'))
    
    # Save encoder for inference
    joblib.dump(role_encoder, "models/job_role_encoder.pkl")
    
    return df, role_encoder

# ----------------------------
# Section 3: Feature Engineering
# ----------------------------
def prepare_features(df):
    """
    Combine text features + numeric structured features + job role encoding
    """
    # Combine text fields
    combined_text = df['Resume'].fillna('') + " " + df['Job Description'].fillna('')
    
    # Only include notable_projects if column exists
    if 'notable_projects' in df.columns:
        combined_text += " " + df['notable_projects'].fillna('').apply(lambda x: " ".join(eval(x)) if x != '' else '')
    
    df['combined_text'] = combined_text
    
    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    X_text = vectorizer.fit_transform(df['combined_text'])
    
    # Numeric structured features
    num_features = np.vstack([
        df['years_experience_numeric'],
        df['education_numeric'],
        df['num_skills'],
        df['num_technologies'],
        df['num_certifications'],
        df['job_role_encoded']
    ]).T
    
    # Combine text + numeric features
    X = hstack([X_text, num_features])
    
    return X, vectorizer

# ----------------------------
# Section 4: Train numeric model
# ----------------------------
def train_numeric_model(df):
    """
    Train multi-output regression model for 7 numeric targets
    """
    X, vectorizer = prepare_features(df)
    
    # Targets: 7 numeric columns
    target_cols = ['experience_match', 'skills_match', 'project_relevance',
                   'tech_match', 'industry_relevance', 'ats_score', 'relevancy']
    y = df[target_cols]
    
    # Split (optional for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and vectorizer
    joblib.dump(model, "models/numeric_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    
    print("Numeric model, vectorizer, and Job Role encoder saved.")
    return model, vectorizer

# ----------------------------
# Section 5: Keyword-based logic
# ----------------------------
def compute_match_scope(resume_text, jd_text):
    resume_keywords = set(extract_keywords(resume_text, top_n=50))
    jd_keywords = set(extract_keywords(jd_text, top_n=50))
    
    match = list(resume_keywords & jd_keywords)
    scope = list(jd_keywords - resume_keywords)
    
    return match, scope

# ----------------------------
# Section 6: Inference function
# ----------------------------
def predict_candidate_score(resume_text, jd_text, job_role, projects_text=""):
    """
    Input: Candidate Resume, Job Description, Job Role, optional Projects
    Output: Dict with 7 numeric scores + match keywords + skill gaps + overall %
    """
    # Load model, vectorizer, and role encoder
    model = joblib.load("models/numeric_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    role_encoder = joblib.load("models/job_role_encoder.pkl")
    
    # Encode job role
    try:
        job_role_encoded = role_encoder.transform([job_role])[0]
    except:
        job_role_encoded = 0  # fallback for unknown roles
    
    # Feature Engineering
    combined_text = clean_text(resume_text) + " " + clean_text(jd_text) + " " + clean_text(projects_text)
    X_text = vectorizer.transform([combined_text])
    
    # Numeric features placeholder (can extend: counts of skills, techs, certifications, experience)
    num_features = np.array([[0, 0, 0, 0, 0, job_role_encoded]])
    
    # Combine text + numeric
    X_input = hstack([X_text, num_features])
    
    # Predict numeric scores
    numeric_cols = ['experience_match', 'skills_match', 'project_relevance',
                    'tech_match', 'industry_relevance', 'ats_score', 'relevancy']
    numeric_preds = model.predict(X_input)[0]
    
    # Compute Match & Scope
    match_keywords, skill_gaps = compute_match_scope(resume_text, jd_text)
    
    # Overall %
    overall_percentage = numeric_preds.sum() / (len(numeric_cols) * 10) * 100
    
    # Build output dict
    output = {col: float(val) for col, val in zip(numeric_cols, numeric_preds)}
    output.update({
        "overall_percentage": overall_percentage,
        "match_keywords": match_keywords,
        "skill_gaps": skill_gaps
    })
    
    return output

# ============================
# End of backend.py
# ============================
