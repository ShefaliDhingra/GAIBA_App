# ----------------------------
# Prediction
# ----------------------------
if st.button("Match"):
    if not resume_text or not jd_text or not job_role:
        st.warning("Please enter Resume, Job Description, and Job Role.")
    else:
        output = predict_candidate_score(model, vectorizer, role_encoder,
                                         resume_text, jd_text, job_role, projects_text)
        st.success("✅ Match Scores")

        # Function to map numeric score to Low/Medium/High
        def score_label(score):
            if score < 8:
                return "Low"
            elif 8 <= score <= 9:
                return "Medium"
            else:
                return "High"

        # Display numeric scores as labels
        st.subheader("Experience Match")
        st.write(score_label(output['experience_match']))

        st.subheader("Skills Match")
        st.write(score_label(output['skills_match']))

        st.subheader("Project Relevance")
        st.write(score_label(output['project_relevance']))

        st.subheader("Tech Match")
        st.write(score_label(output['tech_match']))

        st.subheader("Industry Relevance")
        st.write(score_label(output['industry_relevance']))

        st.subheader("ATS Score")
        st.write(score_label(output['ats_score']))

        st.subheader("Relevancy")
        st.write(score_label(output['relevancy']))

        # Overall match meter (without numeric)
        st.subheader("Overall Match")
        overall = output['overall_percentage']
        if overall < 60:
            label = "Very Low Chances"
            color = "red"
        elif overall < 76:
            label = "Low Chances"
            color = "orange"
        elif overall < 86:
            label = "Medium Chances"
            color = "yellow"
        elif overall < 96:
            label = "High Chances"
            color = "lightgreen"
        else:
            label = "Very High Chances"
            color = "green"
        
        st.progress(int(overall))
        st.markdown(f"**{label}**")

        # Display keywords
        st.subheader("Match Keywords")
        st.write(", ".join(output['match_keywords']))

        st.subheader("Skill / Keyword Gaps")
        st.write(", ".join(output['skill_gaps']))
