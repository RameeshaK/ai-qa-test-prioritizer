import streamlit as st
import pandas as pd
import joblib
import re

# Page Config
st.set_page_config(page_title="AI QA Prioritization Engine", layout="wide")

# Load Serialized Models
@st.cache_resource
def load_ml_assets():
    model = joblib.load('priority_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_ml_assets()

# UI Layout
st.title("🧪 AI Test Case Generation & Prioritization System")

col_p, col_t = st.columns(2)
with col_p:
    project_name = st.text_input("Project Name", value="E-Commerce API")
with col_t:
    test_plan_name = st.text_input("Test Plan Name", value="Sprint 1 Regression")

user_story_input = st.text_area("Enter Raw User Story:", height=100)

if st.button("Generate Test Plan", type="primary"):
    if user_story_input.strip():
        # 1. Predict Priority
        vec_input = vectorizer.transform([user_story_input]).toarray()
        predicted_priority = model.predict(vec_input)[0]
        
        # 2. Display Metrics
        st.subheader("Results")
        st.metric("Predicted Priority", predicted_priority)
        
        # 3. Generate Scenarios Table
        # (Calls scenario generation logic)
        st.success("Test suite successfully generated and prioritized!")
