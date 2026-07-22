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

try:
    model, vectorizer = load_ml_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")

# Dynamic Test Case Engine (Generates All Possible Scenarios)
def generate_all_scenarios(user_story):
    # Extract intent/action keywords
    match = re.search(r"As a (.*?),\s*I want to (.*?)(?:\s*so that (.*))?$", user_story, re.IGNORECASE)
    if match:
        role = match.group(1).strip()
        action = match.group(2).strip()
    else:
        role = "User"
        action = user_story.strip()

    scenarios = []

    # -------------------------------------------------------------
    # 1. POSITIVE TEST SCENARIOS (Valid Flows)
    # -------------------------------------------------------------
    scenarios.extend([
        {
            "Type": "Positive",
            "Scenario": f"Verify successful execution of {action} with valid inputs",
            "Steps": f"1. Log in as {role}\n2. Enter valid required data for {action}\n3. Submit request.",
            "Expected Result": "System processes request successfully and returns confirmation."
        },
        {
            "Type": "Positive",
            "Scenario": f"Verify {action} with optional fields populated",
            "Steps": f"1. Access {action} interface\n2. Fill mandatory AND optional fields with valid data\n3. Click Submit.",
            "Expected Result": "System accepts all data fields correctly without truncation."
        },
        {
            "Type": "Positive",
            "Scenario": f"Verify UI layout and navigation for {action}",
            "Steps": f"1. Navigate to {action} page\n2. Inspect field labels, alignment, and action buttons.",
            "Expected Result": "UI components display properly according to design guidelines."
        }
    ])

    # -------------------------------------------------------------
    # 2. NEGATIVE TEST SCENARIOS (Error Handling & Input Validation)
    # -------------------------------------------------------------
    scenarios.extend([
        {
            "Type": "Negative",
            "Scenario": f"Attempt {action} with all mandatory fields blank",
            "Steps": f"1. Navigate to {action} interface\n2. Leave required fields empty\n3. Click Submit.",
            "Expected Result": "Form submission is blocked; inline validation messages appear."
        },
        {
            "Type": "Negative",
            "Scenario": f"Attempt {action} with invalid input format / special characters",
            "Steps": f"1. Enter unexpected characters (e.g., <script>, sql injection strings) into {action} fields\n2. Submit.",
            "Expected Result": "Input sanitization triggers; system throws error without crashing."
        },
        {
            "Type": "Negative",
            "Scenario": f"Attempt {action} with expired or invalid session token",
            "Steps": f"1. Open {action} interface\n2. Clear session cookies / let session expire\n3. Click Submit.",
            "Expected Result": "System redirects to Login screen with 'Session Expired' notification."
        },
        {
            "Type": "Negative",
            "Scenario": f"Unauthorized execution of {action} (RBAC Check)",
            "Steps": f"1. Log in with an unprivileged role\n2. Attempt to trigger {action} directly via URL or API.",
            "Expected Result": "Access denied with HTTP 403 Forbidden status."
        }
    ])

    # -------------------------------------------------------------
    # 3. EDGE & BOUNDARY SCENARIOS (Stress & Limit Verification)
    # -------------------------------------------------------------
    scenarios.extend([
        {
            "Type": "Edge Case",
            "Scenario": f"Execute {action} with boundary limit input size (Max Length)",
            "Steps": f"1. Enter maximum allowed string length (e.g., 255+ chars) into {action} input fields\n2. Submit.",
            "Expected Result": "System truncates or validates input within character limits safely."
        },
        {
            "Type": "Edge Case",
            "Scenario": f"Execute {action} during sudden network disconnect / high latency",
            "Steps": f"1. Initiate {action}\n2. Simulate network disconnect before server responds.",
            "Expected Result": "System handles timeout gracefully without duplicating records or corrupting database."
        },
        {
            "Type": "Edge Case",
            "Scenario": f"Rapid multi-click / double submission during {action}",
            "Steps": f"1. Enter valid inputs for {action}\n2. Click the Submit button rapidly multiple times.",
            "Expected Result": "Submit button disables on first click; request is processed only once."
        },
        {
            "Type": "Edge Case",
            "Scenario": f"Execute {action} concurrently across multiple tabs",
            "Steps": f"1. Open {action} page in two browser tabs simultaneously\n2. Submit conflicting data from both tabs.",
            "Expected Result": "Concurrency checks prevent race conditions or data overwrite errors."
        }
    ])

    return scenarios

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
        # Predict Priority using Trained Model
        vec_input = vectorizer.transform([user_story_input]).toarray()
        predicted_priority = model.predict(vec_input)[0]

        # Display Metrics
        st.subheader("Results")
        st.metric("Predicted Priority", predicted_priority)
        st.success("Complete test suite generated and prioritized!")

        # Generate All Scenarios
        scenarios = generate_all_scenarios(user_story_input)
        df_results = pd.DataFrame(scenarios)
        df_results['Priority'] = predicted_priority
        df_results['Test Plan'] = test_plan_name

        st.divider()
        st.subheader(f"Generated Test Scenarios & Cases ({len(df_results)} Total)")

        # Filter Tabs for better UX
        tab_all, tab_pos, tab_neg, tab_edge = st.tabs(["All Cases", "Positive", "Negative", "Edge Cases"])
        
        with tab_all:
            st.dataframe(df_results, use_container_width=True)
        with tab_pos:
            st.dataframe(df_results[df_results['Type'] == 'Positive'], use_container_width=True)
        with tab_neg:
            st.dataframe(df_results[df_results['Type'] == 'Negative'], use_container_width=True)
        with tab_edge:
            st.dataframe(df_results[df_results['Type'] == 'Edge Case'], use_container_width=True)

        # Export Option
        csv_data = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Complete Test Suite (CSV)",
            data=csv_data,
            file_name=f"{project_name}_full_test_suite.csv",
            mime="text/csv"
        )
    else:
        st.warning("Please enter a valid user story.")
