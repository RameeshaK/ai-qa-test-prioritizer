import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -------------------------------------------------------------
# 1. PAGE CONFIGURATION & INITIALIZATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Test Case Generation & Prioritization System",
    page_icon="🧪",
    layout="wide"
)

# Load Trained Machine Learning Models
@st.cache_resource
def load_ml_models():
    try:
        model = joblib.load('priority_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model files: {e}. Please ensure 'priority_model.pkl' and 'tfidf_vectorizer.pkl' exist in the repository.")
        return None, None

model, vectorizer = load_ml_models()

# -------------------------------------------------------------
# 2. UI HEADER & INPUT FORM (WITH PLACEHOLDERS)
# -------------------------------------------------------------
st.title("🧪 AI Test Case Generation & Prioritization System")
st.markdown("Automate requirement analysis, risk classification, and prioritized test scenario generation.")

st.divider()

# Project Meta Info (Empty on load, using placeholders)
col1, col2 = st.columns(2)

with col1:
    project_name = st.text_input(
        "Project Name", 
        value="", 
        placeholder="e.g., E-Commerce API"
    )

with col2:
    test_plan_name = st.text_input(
        "Test Plan Name", 
        value="", 
        placeholder="e.g., Sprint 1 Regression"
    )

# Raw User Story Text Input (Empty on load, using placeholders)
user_story_input = st.text_area(
    "Enter Raw User Story:", 
    value="", 
    placeholder="e.g., As a registered customer, I want to process credit card payments securely so that I can complete my order checkout.",
    height=150
)

# -------------------------------------------------------------
# 3. TEST GENERATION & PRIORITIZATION LOGIC
# -------------------------------------------------------------
if st.button("Generate Test Plan", type="primary"):
    # Validation Check
    if not user_story_input.strip():
        st.warning("⚠️ Please enter a user story before generating the test plan.")
    elif model is None or vectorizer is None:
        st.error("❌ Machine Learning model artifacts are not loaded.")
    else:
        with st.spinner("Analyzing requirement, predicting risk priority, and building test suite..."):
            
            # Predict Priority Level using ML Model
            input_vector = vectorizer.transform([user_story_input]).toarray()
            predicted_priority = model.predict(input_vector)[0]
            
            # Get Prediction Probabilities
            probabilities = model.predict_proba(input_vector)[0]
            classes = model.classes_
            prob_dict = {cls: round(prob * 100, 2) for cls, prob in zip(classes, probabilities)}

            # Display Priority Status Badge
            st.divider()
            st.subheader("📌 Requirement Priority Analysis")
            
            badge_color = {
                "High": "🔴 **HIGH PRIORITY** (Critical Path / High Risk)",
                "Medium": "🟡 **MEDIUM PRIORITY** (Core Functional Path)",
                "Low": "🟢 **LOW PRIORITY** (UI / Cosmetic Path)"
            }
            
            st.markdown(f"**Predicted Execution Priority:** {badge_color.get(predicted_priority, predicted_priority)}")
            
            # Show Probability breakdown in expander
            with st.expander("View Confidence Breakdown"):
                st.json(prob_dict)

            # Generate Test Cases Matrix
            st.subheader("📋 Prioritized Test Case Suite")

            test_cases = [
                {
                    "Test ID": "TC-001",
                    "Scenario": "Positive Functional Flow",
                    "Description": f"Verify successful execution for: '{user_story_input[:60]}...'",
                    "Expected Result": "System processes requirement successfully with 200 OK status.",
                    "Priority": predicted_priority,
                    "Execution Order": "1 (Run First)" if predicted_priority == "High" else "2"
                },
                {
                    "Test ID": "TC-002",
                    "Scenario": "Negative / Boundary Flow",
                    "Description": "Verify system handling with invalid inputs, missing fields, or empty parameters.",
                    "Expected Result": "System gracefully handles error and displays appropriate validation prompt.",
                    "Priority": predicted_priority,
                    "Execution Order": "2" if predicted_priority == "High" else "3"
                },
                {
                    "Test ID": "TC-003",
                    "Scenario": "Security & Authorization Check",
                    "Description": "Verify endpoint security, session expiration, and unauthorized access constraints.",
                    "Expected Result": "Unauthorized requests are rejected with 401/403 HTTP status.",
                    "Priority": "High",
                    "Execution Order": "1 (Run First)"
                },
                {
                    "Test ID": "TC-004",
                    "Scenario": "UI Alignment & Responsiveness",
                    "Description": "Verify UI layout rendering across mobile, desktop, and dark mode themes.",
                    "Expected Result": "UI components align properly with correct visual styling.",
                    "Priority": "Low",
                    "Execution Order": "4 (Run Last)"
                }
            ]

            df_tests = pd.DataFrame(test_cases)
            
            # Render Styled Table
            st.dataframe(df_tests, use_container_width=True, hide_index=True)

            # Export Button
            csv_data = df_tests.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Test Plan to CSV",
                data=csv_data,
                file_name=f"test_plan_{project_name.lower().replace(' ', '_') or 'export'}.csv",
                mime="text/csv"
            )
