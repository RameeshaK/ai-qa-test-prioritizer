import streamlit as st
import joblib
import pandas as pd
import json
import os

# -------------------------------------------------------------
# 1. PAGE CONFIGURATION & SESSION STATE SETUP
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Test Case Generation & Prioritization System",
    page_icon="🧪",
    layout="wide"
)

# Dummy User Database (You can replace this with a real DB or JSON file later)
USER_DB = {
    "qa_lead": "password123",
    "tester1": "qa2026"
}

USER_HISTORY_FILE = "user_history.json"

# Initialize Session States
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "current_test_cases" not in st.session_state:
    st.session_state.current_test_cases = None

# Helper functions to persist user history to JSON
def load_all_history():
    if os.path.exists(USER_HISTORY_FILE):
        with open(USER_HISTORY_FILE, "r") as f:
            return json.load(f)
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

# Load ML Models
@st.cache_resource
def load_ml_models():
    try:
        model = joblib.load('priority_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, vectorizer = load_ml_models()

# -------------------------------------------------------------
# 2. SIDEBAR - LOGIN / LOGOUT MANAGEMENT
# -------------------------------------------------------------
st.sidebar.title("👤 User Authentication")

if not st.session_state.authenticated:
    st.sidebar.subheader("Login for QA Engineers")
    login_user = st.sidebar.text_input("Username", value="", placeholder="e.g., qa_lead")
    login_pass = st.sidebar.text_input("Password", type="password", value="", placeholder="••••••••")
    
    if st.sidebar.button("Login", type="primary"):
        if login_user in USER_DB and USER_DB[login_user] == login_pass:
            st.session_state.authenticated = True
            st.session_state.username = login_user
            st.sidebar.success(f"Logged in as {login_user}")
            st.rerun()
        else:
            st.sidebar.error("Invalid Username or Password")
    st.sidebar.info("💡 **Guest Mode:** You can generate test cases without logging in, but reloading will clear your data.")
else:
    st.sidebar.success(f"Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.current_test_cases = None
        st.rerun()

# -------------------------------------------------------------
# 3. MAIN APP HEADER & INPUT FORM
# -------------------------------------------------------------
st.title("🧪 AI Test Case Generation & Prioritization System")
st.markdown("Automate requirement analysis, risk classification, and prioritized test scenario generation.")

st.divider()

col1, col2 = st.columns(2)
with col1:
    project_name = st.text_input("Project Name", value="", placeholder="e.g., E-Commerce API")
with col2:
    test_plan_name = st.text_input("Test Plan Name", value="", placeholder="e.g., Sprint 1 Regression")

user_story_input = st.text_area(
    "Enter Raw User Story:", 
    value="", 
    placeholder="e.g., As a customer, I want to process credit card payments securely so that I can complete my checkout.",
    height=120
)

# -------------------------------------------------------------
# 4. GENERATE TEST CASES & PERSISTENCE LOGIC
# -------------------------------------------------------------
if st.button("Generate Test Plan", type="primary"):
    if not user_story_input.strip():
        st.warning("⚠️ Please enter a user story before generating the test plan.")
    elif model is None or vectorizer is None:
        st.error("❌ Machine Learning model artifacts are not loaded.")
    else:
        with st.spinner("Analyzing requirement and predicting execution priority..."):
            input_vector = vectorizer.transform([user_story_input]).toarray()
            predicted_priority = model.predict(input_vector)[0]
            
            test_cases = [
                {
                    "Test ID": "TC-001",
                    "Scenario": "Positive Functional Flow",
                    "Description": f"Verify successful execution for: '{user_story_input[:50]}...'",
                    "Expected Result": "System processes request with 200 OK status.",
                    "Priority": predicted_priority,
                    "Execution Order": "1 (Run First)" if predicted_priority == "High" else "2"
                },
                {
                    "Test ID": "TC-002",
                    "Scenario": "Security Check",
                    "Description": "Verify authentication constraints and data encryption.",
                    "Expected Result": "Unauthorized access is blocked.",
                    "Priority": "High",
                    "Execution Order": "1 (Run First)"
                }
            ]
            
            st.session_state.current_test_cases = {
                "project": project_name or "Untitled Project",
                "test_plan": test_plan_name or "Untitled Plan",
                "story": user_story_input,
                "priority": predicted_priority,
                "cases": test_cases
            }
            
            # Save to permanent JSON store ONLY if user is logged in
            if st.session_state.authenticated:
                save_user_task(st.session_state.username, st.session_state.current_test_cases)
                st.success("✅ Test plan generated and saved to your account!")
            else:
                st.info("ℹ️ Test plan generated in Guest Mode. Reloading the page will discard this result.")

# Render Current Generated Output
if st.session_state.current_test_cases:
    st.divider()
    st.subheader(f"📌 Current Output: {st.session_state.current_test_cases['project']}")
    df_current = pd.DataFrame(st.session_state.current_test_cases['cases'])
    st.dataframe(df_current, use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# 5. PREVIOUS TASKS SECTION (LOGGED-IN USERS ONLY)
# -------------------------------------------------------------
if st.session_state.authenticated:
    st.divider()
    st.subheader("📚 Saved Task History")
    
    saved_tasks = get_user_tasks(st.session_state.username)
    
    if saved_tasks:
        for idx, task in enumerate(reversed(saved_tasks)):
            with st.expander(f"📁 {task['project']} - {task['test_plan']} (Priority: {task['priority']})"):
                st.write(f"**User Story:** {task['story']}")
                st.dataframe(pd.DataFrame(task['cases']), use_container_width=True, hide_index=True)
    else:
        st.write("No saved tasks found. Generate a test plan above to save it to your history.")
