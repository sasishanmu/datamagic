import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Data Transformer",
    page_icon="✨",
    layout="wide"
)

# --- SECURELY CONFIGURE GOOGLE API KEY ---
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])


# --- AI FUNCTION ---
def get_ai_command(user_query, df_head):
    model = genai.GenerativeModel('gemini-pro-latest')
    prompt = f"""
    You are an expert Python data scientist.
    Given a pandas DataFrame named `df` with these first few rows:
    {df_head}

    Write a single, executable line of Python code to perform the following task:
    '{user_query}'

    The code must manipulate the DataFrame `df` directly. Do not add any explanation, comments, or markdown formatting.
    Only return the raw Python code.
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip().replace("```python", "").replace("```", "").strip()
    except Exception as e:
        print(f"Error calling Google Gemini API: {e}")
        return None


# --- APP TITLE AND DESCRIPTION ---
st.title("🤖 DataMagic-AI Powered: Your data, magically transformed!")
st.write("Welcome! Upload a CSV file and use plain English to clean, modify, and transform your data instantly.")

# --- FILE UPLOADER AND INITIALIZATION ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    if 'df_original' not in st.session_state or st.session_state.file_id != uploaded_file.name:
        st.session_state.file_id = uploaded_file.name
        st.session_state.df_original = pd.read_csv(uploaded_file)
        st.session_state.df_transformed = st.session_state.df_original.copy()
        st.session_state.transformations = []

    # --- TRANSFORMATION CONTROLS ---
    st.subheader("Transform Your Data ✨")
    user_command = st.text_input("Enter your command (e.g., 'drop rows where age is less than 18')", key="user_command")

    col_run, col_reset = st.columns([1, 5])

    with col_run:
        if st.button("Apply Changes", type="primary"):
            if user_command:
                df_head_str = str(st.session_state.df_original.head().to_dict())
                ai_code = get_ai_command(user_command, df_head_str)

                if ai_code:
                    with st.expander("View Executed Python Code"):
                        st.code(ai_code, language='python')

                    try:
                        rows_before = len(st.session_state.df_transformed)

                        exec_scope = {'df': st.session_state.df_transformed, 'pd': pd , 'np': np}
                        exec(ai_code, exec_scope)
                        st.session_state.df_transformed = exec_scope['df']

                        rows_after = len(st.session_state.df_transformed)

                        transformation_log = {
                            "step": len(st.session_state.transformations) + 1,
                            "description": user_command,
                            "rows_affected": rows_after - rows_before
                        }
                        st.session_state.transformations.append(transformation_log)
                        st.success("Transformation applied successfully!")

                    except Exception as e:
                        st.error(f"Oops! An error occurred: {e}")
                else:
                    st.warning("Could not generate a command.")
            else:
                st.warning("Please enter a command.")

    with col_reset:
        if st.button("Reset Data"):
            st.session_state.df_transformed = st.session_state.df_original.copy()
            st.session_state.transformations = []
            st.success("Data has been reset to its original state.")

    # --- SUMMARY SECTION ---
    st.subheader("Summary")

    sum_col1, sum_col2 = st.columns(2)

    with sum_col1:
        st.metric(label="Total Transformations Applied", value=len(st.session_state.transformations))

    with sum_col2:
        original_rows = len(st.session_state.df_original)
        transformed_rows = len(st.session_state.df_transformed)
        st.metric(label="Current Row Count", value=transformed_rows, delta=transformed_rows - original_rows)

    st.markdown("---")

    if not st.session_state.transformations:
        st.write("No transformations have been applied yet.")
    else:
        summary_df = pd.DataFrame(st.session_state.transformations)
        summary_df.rename(columns={
            "step": "Step #",
            "description": "Description",
            "rows_affected": "Rows Affected"
        }, inplace=True)

        # --- MODIFIED: Ensure the summary table uses the full container width ---
        st.dataframe(summary_df, use_container_width=True)

    st.markdown("---")

    # --- SIDE-BY-SIDE DATA DISPLAY ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Data")
        st.dataframe(st.session_state.df_original, height=400)

    with col2:
        st.subheader("Transformed Data")
        st.dataframe(st.session_state.df_transformed, height=400)

        csv = st.session_state.df_transformed.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Transformed CSV",
            data=csv,
            file_name='transformed_data.csv',
            mime='text/csv',
        )
else:
    st.info("Upload a CSV file to get started.")
    with st.expander("How it works"):
        st.write("""
            1.  **Upload:** Drop any CSV file into the uploader.
            2.  **Command:** Type a data cleaning instruction in plain English.
            3.  **Apply:** Hit "Apply Changes". Our AI will translate your command and execute it.
            4.  **View & Download:** Instantly see the result and download your new data.
        """)