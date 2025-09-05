# ----------------------------
# Prediction
# ----------------------------
if st.button("Match"):
    if not resume_text or not jd_text or not job_role:
        st.warning("Please enter Resume, Job Description, and Job Role.")
    else:
        output = predict_candidate_score(
            model, vectorizer, role_encoder,
            resume_text, jd_text, job_role, projects_text
        )

        # ------------------------
        # Current Resume Strength (Keyword-based score)
        # ------------------------
        st.subheader("📌 Current Resume Strength")
        keyword_count = len(output['match_keywords'])
        if keyword_count == 0:
            resume_strength = "Low"
        elif 1 <= keyword_count <= 3:
            resume_strength = "Medium"
        else:
            resume_strength = "High"
        st.write(f"**{resume_strength}** (based on {keyword_count} matched keywords)")

        # ------------------------
        # Display keywords nicely
        # ------------------------
        st.subheader("Match Keywords")
        if output['match_keywords']:
            st.markdown(" ".join([f"✅ `{kw}`" for kw in output['match_keywords']]))
        else:
            st.write("No significant match keywords found.")

        st.subheader("Skill / Keyword Gaps")
        if output['skill_gaps']:
            st.markdown(" ".join([f"⚠️ `{kw}`" for kw in output['skill_gaps']]))
        else:
            st.write("No significant skill gaps found.")

        # ------------------------
        # Click-to-expand detailed scores
        # ------------------------
        with st.expander("📊 Detailed Scores"):
            scores = [
                'experience_match', 'skills_match', 'project_relevance',
                'tech_match', 'industry_relevance', 'ats_score', 'relevancy'
            ]
            cols = st.columns(4)
            for idx, score_name in enumerate(scores):
                col = cols[idx % 4]
                col.metric(
                    label=score_name.replace("_", " ").title(),
                    value=f"{output[score_name]:.1f}/10"
                )
