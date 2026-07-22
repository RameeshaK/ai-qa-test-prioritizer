import streamlit as st
import pandas as pd
import joblib
import re

# Page Configuration
st.set_page_config(
    page_title="AI Test Case Generation & Prioritization System",
    page_icon="🧪",
    layout="wide"
)

# Load Serialized Models
@st.cache_resource
def load_ml_assets():
    model = joblib.load('priority_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

# Load ML components
try:
    model, vectorizer = load_ml_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")

# Function to generate scenarios
def generate_scenarios(user_story):
    match = re.search(r"As a (.*?),\s*I want to (.*?)(?:\s*so that (.*))?$", user_story, re.IGNORECASE)
    if match:
        role = match.group(1).strip()
        action = match.group(2).strip()
    else:
        role = "User"
        action = user_story.strip()

    return [
        {
            "Type": "Positive",
            "Scenario": f"Verify successful execution of {action}",
            "Steps": f"1. Log in as {role}\n2. Perform action: {action}\n3. Confirm valid response.",
            "Expected Result": "Action completed successfully with expected state."
        },
        {
            "Type": "Negative",
            "Scenario": f"Attempt {action} with missing required fields",
            "Steps": f"1. Navigate to {action} interface\n2. Leave mandatory fields blank\n3. Click submit.",
            "Expected Result": "System prevents process and displays validation error message."
        },
        {
            "Type": "Edge Case",
            "Scenario": f"Execute {action} during network interruption",
            "Steps": f"1. Initiate {action}\n2. Disconnect network mid-process.",
            "Expected Result": "System handles timeout gracefully without duplicating records."
        }
    ]

# --- UI LAYOUT ---
st.title("🧪 AI Test Case Generation & Prioritization System")

col1, col2 = st.columns(2)
with col1:
    project_name = st.text_input("Project Name", value="E-Commerce API")
with col2:
    test_plan_name = st.text_input("Test Plan Name", value="Sprint 1 Regression")

user_story_input = st.text_area(
    "Enter Raw User Story:",
    value="user should be able to login to home using login interface",
    height=100
)

if st.button("Generate Test Plan", type="primary"):
    if user_story_input.strip():
        # Predict Priority
        vec_input = vectorizer.transform([user_story_input]).toarray()
        predicted_priority = model.predict(vec_input)[0]

        # Display Metrics
        st.subheader("Results")
        st.metric("Predicted Priority", predicted_priority)
        st.success("Test suite successfully generated and prioritized!")

        # Generate & Display Table
        scenarios = generate_scenarios(user_story_input)
        df_results = pd.DataFrame(scenarios)
        df_results['Priority'] = predicted_priority
        df_results['Test Plan'] = test_plan_name

        st.divider()
        st.subheader("Generated Test Scenarios & Cases")
        st.dataframe(df_results, use_container_width=True)

        # Download Button
        csv_data = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Test Suite (CSV)",
            data=csv_data,
            file_name=f"{project_name}_test_plan.csv",
            mime="text/csv"
        )
    else:
        st.warning("Please enter a valid user story.")
