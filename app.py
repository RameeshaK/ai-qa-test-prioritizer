import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import os

# -------------------------------------------------------------
# 1. PAGE CONFIGURATION & INITIALIZATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Test Case Generation & Prioritization System",
    page_icon="🧪",
    layout="wide"
)

# Dummy User Credentials Database (Expand or replace with a real database as needed)
USER_DB = {
    "qa_lead": "password123",
    "tester1": "qa2026"
}

USER_HISTORY_FILE = "user_history.json"

# Initialize Session State Keys
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "guest_test_plan" not in st.session_state:
    st.session_state.guest_test_plan = None

# History File Helper Functions
def load_all_history():
    if os.path.exists(USER_HISTORY_FILE):
        try:
            with open(USER_HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_user_task(username, task_data):
    history = load_all_history()
    if username not in history:
        history[username] = []
    history[username].append(task_data)
    with open(USER_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def get_user_tasks(username):
    history = load_all_history()
    return history.get(username, [])

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
# 2. SIDEBAR - USER AUTHENTICATION
# -------------------------------------------------------------
st.sidebar.title("🔐 User Authentication")

if not st.session_state.authenticated:
    st.sidebar.subheader("QA Engineer Login")
    login_user = st.sidebar.text_input("Username", value="", placeholder="e.g., qa_lead")
    login_pass = st.sidebar.text_input("Password", type="password", value="", placeholder="••••••••")
    
    if st.sidebar.button("Login", type="primary"):
        if login_user in USER_DB and USER_DB[login_user] == login_pass:
            st.session_state.authenticated = True
            st.session_state.username = login_user
            st.session_state.guest_test_plan = None  # Clear temporary guest data on login
            st.sidebar.success(f"Logged in as {login_user}")
            st.rerun()
        else:
            st.sidebar.error("Invalid Username or Password")
    
    st.sidebar.info("💡 **Guest Mode active:** You can generate test plans without logging in, but reloading the page will erase your output.")
else:
    st.sidebar.success(f"Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.guest_test_plan = None
        st.rerun()

# -------------------------------------------------------------
# 3. UI HEADER & INPUT FORM (WITH PLACEHOLDERS)
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
# 4. TEST GENERATION & PRIORITIZATION LOGIC
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

            # Generate Test Cases Matrix
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

            # Construct Current Task Payload
            task_payload = {
                "project_name": project_name or "Untitled Project",
                "test_plan_name": test_plan_name or "Untitled Plan",
                "user_story": user_story_input,
                "predicted_priority": predicted_priority,
                "prob_dict": prob_dict,
                "test_cases": test_cases
            }

            # Save to JSON storage if user is logged in
            if st.session_state.authenticated:
                save_user_task(st.session_state.username, task_payload)
                st.success("✅ Test plan generated and saved to your account history!")
            else:
                st.session_state.guest_test_plan = task_payload
                st.info("ℹ️ Generated in Guest Mode. Reloading the page will clear these test cases.")

# -------------------------------------------------------------
# 5. DISPLAY CURRENT GENERATED TEST PLAN
# -------------------------------------------------------------
current_plan = None

# If user is logged in, show their latest generated task (or from session)
if st.session_state.authenticated:
    user_tasks = get_user_tasks(st.session_state.username)
    if user_tasks:
        current_plan = user_tasks[-1]
else:
    current_plan = st.session_state.guest_test_plan

if current_plan:
    st.divider()
    st.subheader("📌 Requirement Priority Analysis")
    
    badge_color = {
        "High": "🔴 **HIGH PRIORITY** (Critical Path / High Risk)",
        "Medium": "🟡 **MEDIUM PRIORITY** (Core Functional Path)",
        "Low": "🟢 **LOW PRIORITY** (UI / Cosmetic Path)"
    }
    
    st.markdown(f"**Predicted Execution Priority:** {badge_color.get(current_plan['predicted_priority'], current_plan['predicted_priority'])}")
    
    # Show Probability breakdown in expander
    with st.expander("View Confidence Breakdown"):
        st.json(current_plan['prob_dict'])

    # Generate Test Cases Matrix
    st.subheader(f"📋 Prioritized Test Case Suite ({current_plan['project_name']} - {current_plan['test_plan_name']})")

    df_tests = pd.DataFrame(current_plan['test_cases'])
    
    # Render Styled Table
    st.dataframe(df_tests, use_container_width=True, hide_index=True)

    # Export Button
    csv_data = df_tests.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Test Plan to CSV",
        data=csv_data,
        file_name=f"test_plan_{current_plan['project_name'].lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

# -------------------------------------------------------------
# 6. HISTORICAL TASKS SECTION (LOGGED-IN USERS ONLY)
# -------------------------------------------------------------
if st.session_state.authenticated:
    st.divider()
    st.subheader("📚 Saved Task History")
    
    user_tasks = get_user_tasks(st.session_state.username)
    
    if user_tasks:
        for idx, task in enumerate(reversed(user_tasks)):
            with st.expander(f"📁 {task['project_name']} — {task['test_plan_name']} (Priority: {task['predicted_priority']})"):
                st.write(f"**User Story:** {task['user_story']}")
                st.dataframe(pd.DataFrame(task['test_cases']), use_container_width=True, hide_index=True)
    else:
        st.info("No saved tasks found. Generate a test plan above to store it in your history.")
