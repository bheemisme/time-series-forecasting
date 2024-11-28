import streamlit as st

from src.pages import yahoo

# A dictionary to map page names to their respective modules
PAGES = {
    "yahoo": yahoo,
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# initialising the app
if selection is not None:
    page = PAGES[selection]
    page.app()