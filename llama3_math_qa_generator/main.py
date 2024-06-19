import streamlit as st
import pandas as pd
import groq_api


col1, col2 = st.columns([2,3])

math_qa_df = pd.DataFrame({
            "IB-Math": ["Question", "Workings + Answer"],
            "Generated": ["", ""]
        })

with col1:
    st.title("Llama 3: IB Math (HL/SL) Q&A Generator")
    math_context = st.text_area("Paste the relevant topic context: ", height=300)
    if st.button("Generate"):
        math_qa_df = groq_api.retrieve_QA_from_context(math_context)

with col2:
    st.markdown("<br/>" * 5, unsafe_allow_html=True)  # Creates 5 lines of vertical space
    st.table(
        math_qa_df
    )