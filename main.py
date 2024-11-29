import streamlit as st

from src.pages import yahoo, gold, apple, sales, climate

PAGES = {
    "yahoo": yahoo,
    "gold": gold,
    "apple": apple,
    "sales": sales,
    "climate": climate
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

if selection is not None:
    page = PAGES[selection]
    page.app()

