import streamlit as st
import pandas as pd
import joblib
import re
import json
import os

# Page Configuration
st.set_page_config(
    page_title="AI Test Case Generation & Prioritization System",
    page_icon="🧪",
    layout="wide"
)

# -------------------------------------------------------------
# USER AUTHENTICATION & PERSISTENCE SETUP
# -------------------------------------------------------------
USER_DB = {
    "qa_lead": "password123",
    "tester1": "qa2026"
}

USER_HISTORY_FILE = "user_history.json"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "guest_results" not in st.session_state:
    st.session_state.guest_results = None

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

# --- SIDEBAR AUTHENTICATION ---
st.sidebar.title("🔐 User Authentication")

if not st.session_state.authenticated:
    st.sidebar.subheader("QA Engineer Login")
    login_user = st.sidebar.text_input("Username", value="", placeholder="e.g., qa_lead")
    login_pass = st.sidebar.text_input("Password", type="password", value="", placeholder="••••••••")
    
    if st.sidebar.button("Login", type="primary"):
        if login_user in USER_DB and USER_DB[login_user] == login_pass:
            st.session_state.authenticated = True
            st.session_state.username = login_user
            st.session_state.guest_results = None
            st.sidebar.success(f"Logged in as {login_user}")
            st.rerun()
        else:
            st.sidebar.error("Invalid Username or Password")
            
    st.sidebar.info("💡 **Guest Mode:** Generating without login will clear test cases on page reload.")
else:
    st.sidebar.success(f"Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.guest_results = None
        st.rerun()

# --- UI LAYOUT ---
st.title("🧪 AI Test Case Generation & Prioritization System")

col1, col2 = st.columns(2)
with col1:
    project_name = st.text_input("Project Name", value="", placeholder="e.g., E-Commerce API")
with col2:
    test_plan_name = st.text_input("Test Plan Name", value="", placeholder="e.g., Sprint 1 Regression")

user_story_input = st.text_area(
    "Enter Raw User Story:",
    value="",
    placeholder="e.g., user should be able to login to home using login interface",
    height=100
)

if st.button("Generate Test Plan", type="primary"):
    if user_story_input.strip():
        # Predict Priority using Trained Model
        vec_input = vectorizer.transform([user_story_input]).toarray()
        predicted_priority = model.predict(vec_input)[0]

        # Generate All Scenarios
        scenarios = generate_all_scenarios(user_story_input)
        
        task_data = {
            "project_name": project_name or "E-Commerce API",
            "test_plan_name": test_plan_name or "Sprint 1 Regression",
            "user_story": user_story_input,
            "predicted_priority": predicted_priority,
            "scenarios": scenarios
        }

        if st.session_state.authenticated:
            save_user_task(st.session_state.username, task_data)
            st.success("✅ Complete test suite generated and saved to your account!")
        else:
            st.session_state.guest_results = task_data
            st.info("ℹ️ Generated in Guest Mode. Reloading the page will clear these results.")
    else:
        st.warning("Please enter a valid user story.")

# Render Current Task Output
active_data = None

if st.session_state.authenticated:
    user_tasks = get_user_tasks(st.session_state.username)
    if user_tasks:
        active_data = user_tasks[-1]
else:
    active_data = st.session_state.guest_results

if active_data:
    st.subheader("Results")
    st.metric("Predicted Priority", active_data["predicted_priority"])

    df_results = pd.DataFrame(active_data["scenarios"])
    df_results['Priority'] = active_data["predicted_priority"]
    df_results['Test Plan'] = active_data["test_plan_name"]

    st.divider()

    # -------------------------------------------------------------
    # 1. INITIAL PRIORITY SUMMARY TABLE (Type, Scenario, Priority)
    # -------------------------------------------------------------
    st.subheader("📌 Scenario Priority Summary")
    
    # Helper to add colored priority badges
    def format_priority(val):
        if val == "High":
            return "🔴 High"
        elif val == "Medium":
            return "🟡 Medium"
        elif val == "Low":
            return "🟢 Low"
        return val

    df_summary = df_results[['Type', 'Scenario', 'Priority']].copy()
    df_summary['Priority'] = df_summary['Priority'].apply(format_priority)

    st.dataframe(
        df_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Scenario": st.column_config.TextColumn("Scenario", width="large"),
            "Priority": st.column_config.TextColumn("Priority", width="small"),
        }
    )

    # -------------------------------------------------------------
    # 2. SELECTABLE SCENARIO DETAILED VIEW
    # -------------------------------------------------------------
    st.divider()
    st.subheader("🔍 Scenario Detail Inspector")
    
    selected_scenario_name = st.selectbox(
        "Select a test scenario to inspect full details:",
        options=df_results['Scenario'].tolist()
    )

    if selected_scenario_name:
        selected_row = df_results[df_results['Scenario'] == selected_scenario_name].iloc[0]
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.write(f"**Type:** {selected_row['Type']}")
        with col_b:
            st.write(f"**Priority:** {format_priority(selected_row['Priority'])}")
        with col_c:
            st.write(f"**Test Plan:** {selected_row['Test Plan']}")
            
        st.write(f"**Scenario:** {selected_row['Scenario']}")
        st.info(f"**Steps:**\n{selected_row['Steps']}")
        st.success(f"**Expected Result:**\n{selected_row['Expected Result']}")

    # -------------------------------------------------------------
    # 3. COMPLETE TEST SUITE TABBED MATRIX
    # -------------------------------------------------------------
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
        file_name=f"{active_data['project_name']}_full_test_suite.csv",
        mime="text/csv"
    )

# --- HISTORICAL TASKS FOR LOGGED-IN USERS ---
if st.session_state.authenticated:
    st.divider()
    st.subheader("📚 Saved Task History")
    
    saved_tasks = get_user_tasks(st.session_state.username)
    
    if saved_tasks:
        for idx, task in enumerate(reversed(saved_tasks)):
            with st.expander(f"📁 {task['project_name']} — {task['test_plan_name']} (Priority: {task['predicted_priority']})"):
                st.write(f"**User Story:** {task['user_story']}")
                df_hist = pd.DataFrame(task['scenarios'])
                df_hist['Priority'] = task['predicted_priority']
                df_hist['Test Plan'] = task['test_plan_name']
                st.dataframe(df_hist, use_container_width=True)
    else:
        st.info("No saved tasks found. Generate a test plan above to save it to your history.")
