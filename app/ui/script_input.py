import streamlit as st
from app.storage import state

def script_input():
    st.header("2. Upload Rewritten Dream Script")
    uploaded = st.file_uploader("Upload rewritten_script.txt", type=["txt"], key="script_upload")
    if uploaded:
        text = uploaded.read().decode("utf-8").strip()
        if len(text) < 40:
            st.error("Script is too short. Please provide a detailed rewritten dream.")
            return
        state.save_script(text)
        st.success("Script uploaded.")
        st.text_area("Script Preview", text, height=200, disabled=True)
    else:
        st.info("Upload your calm, rewritten dream script as a .txt file.")
