import streamlit as st
import pandas as pd
import joblib

# (Keep your existing page config, model loading, and generator logic here)

# ...

if st.button("Generate Test Plan", type="primary"):
    if user_story_input.strip():
        # 1. Predict Priority using ML Model
        vec_input = vectorizer.transform([user_story_input]).toarray()
        predicted_priority = model.predict(vec_input)[0]
        
        # 2. Display Metrics & Status
        st.subheader("Results")
        st.metric("Predicted Priority", predicted_priority)
        st.success("Test suite successfully generated and prioritized!")
        
        # 3. Generate Scenarios and Test Cases
        # (Calls the scenario generator function from Phase 3)
        scenarios = generate_scenarios(user_story_input)
        
        # Convert list of scenarios to Pandas DataFrame
        df_results = pd.DataFrame(scenarios)
        df_results['Priority'] = predicted_priority
        df_results['Test Plan'] = test_plan_name
        
        st.divider()
        
        # 4. Render Table View on Screen
        st.subheader("Generated Test Scenarios & Cases")
        st.dataframe(
            df_results[['Type', 'Scenario', 'Steps', 'Expected Result', 'Priority', 'Test Plan']],
            use_container_width=True
        )
        
        # 5. Add Download / Export Option
        csv_data = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Test Suite (CSV)",
            data=csv_data,
            file_name=f"{project_name}_test_plan.csv",
            mime="text/csv"
        )
    else:
        st.warning("Please enter a valid user story before generating.")
