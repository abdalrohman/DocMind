import streamlit as st


def disable_deploy():
    # Remove the Streamlit `Deploy` button from the Header
    st.markdown(
        r"""
    <style>
    .stDeployButton {
            visibility: hidden;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
