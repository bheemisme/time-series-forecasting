import streamlit as st

from src.pages import yahoo, gold, apple

# A dictionary to map page names to their respective modules
PAGES = {
    "yahoo": yahoo,
    "gold": gold,
    "apple": apple
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# initialising the app
if selection is not None:
    page = PAGES[selection]
    page.app()