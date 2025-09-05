def predict_candidate_score(model, vectorizer, role_encoder,
                            resume_text, jd_text, job_role, projects_text="",
                            years_experience=None, education_level=None):
    
    # Encode job role
    try:
        job_role_encoded = role_encoder.transform([job_role])[0]
    except:
        job_role_encoded = 0
    
    # Use numeric features if provided
    years_experience_numeric = map_experience_to_numeric(years_experience) if years_experience else 0
    education_numeric = map_education_to_numeric(education_level) if education_level else 0
    
    # TF-IDF features
    combined_text = clean_text(resume_text) + " " + clean_text(jd_text) + " " + clean_text(projects_text)
    X_text = vectorizer.transform([combined_text])
    
    # Numeric features
    num_features = np.array([[years_experience_numeric, education_numeric]])
    X_input = hstack([X_text, num_features])
    
    numeric_cols = ['experience_match','skills_match','project_relevance',
                    'tech_match','industry_relevance','ats_score','relevancy']
    numeric_preds = model.predict(X_input)[0]
    
    # Keyword match
    match_keywords, skill_gaps = compute_match_scope(resume_text, jd_text)
    
    # -------------------------
    # Base overall score (from ML)
    # -------------------------
    ml_score = numeric_preds.sum() / (len(numeric_cols)*10) * 100

    # -------------------------
    # Stronger Keyword Penalty
    # -------------------------
    jd_keywords = set(extract_keywords(jd_text, top_n=30))
    overlap_ratio = len(match_keywords) / (len(jd_keywords) + 1e-6)

    # Weight keywords heavily
    keyword_score = overlap_ratio * 100  

    # Combine ML + keyword overlap
    # (70% weight keywords, 30% weight ML)
    overall_percentage = (0.3 * ml_score) + (0.7 * keyword_score)

    # Force lower bound if no overlap at all
    if overlap_ratio == 0:
        overall_percentage = min(overall_percentage, 20)

    # -------------------------
    # Prepare output
    # -------------------------
    output = {col: float(val) for col, val in zip(numeric_cols, numeric_preds)}
    output.update({
        "overall_percentage": overall_percentage,
        "match_keywords": match_keywords,
        "skill_gaps": skill_gaps
    })
    
    return output

