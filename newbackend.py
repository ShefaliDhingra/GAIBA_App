# backend.py (simplified)

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# ----------------------------
# Utility functions
# ----------------------------
def map_experience_to_numeric(experience_str):
    mapping = {"entry-level":1,"mid-level":5,"senior-level":8,"expert":10}
    return mapping.get(str(experience_str).lower(),0)

def map_education_to_numeric(education_str):
    mapping = {"bachelors degree":1,"masters degree":2,"phd":3}
    return mapping.get(str(education_str).lower(),0)

# ----------------------------
# Load data
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
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = lgb.LGBMRegressor(objective='regression', n_estimators=500, learning_rate=0.05)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], e_
