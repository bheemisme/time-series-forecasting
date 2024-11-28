import streamlit as st
import numpy as np
from src.models.gold import model
from src.models.utils import lstm_forecast, xgb_forecast


def app():
    st.title('Time Series Forecasting')
    st.markdown('## Gold ETF Forecasting')

    user_input = st.text_input("Enter numbers separated by commas:")

    if user_input:
        try:
            numbers = [float(num.strip()) for num in user_input.split(",")]
            
            
            numbers = np.array(numbers)
            fig1 = lstm_forecast(model.gold_lstm_model, numbers,'Close', model.gold_scaler)
            fig2 = xgb_forecast(model.gold_xgb_model,numbers,'Close', model.gold_scaler)

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig1)
            with col2:
                st.pyplot(fig2)
        except ValueError:
            st.error("Please enter valid numbers separated by commas.")
    