import streamlit as st

# Hiding the default sidebar menu
st.markdown(
    """
    <style>
        [data-testid="stSidebarNavItems"] {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True
)

# Sidebar links
with st.sidebar:
    st.write("Navigation")
    st.page_link("app7.py", label="Home")
    st.page_link("pages/profile.py", label="Profile")

# Main page content
st.write("Hello")
