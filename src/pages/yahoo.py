import streamlit as st
import numpy as np
from src.models.yahoo import model
from src.models.utils import lstm_forecast, xgb_forecast


def app():
    st.title('Time Series Forecasting')
    st.markdown('## Yahoo Stock Forecasting')

    user_input = st.text_input("Enter numbers separated by commas:")

    if user_input:
        try:
            numbers = [float(num.strip()) for num in user_input.split(",")]
            
            fig1 = lstm_forecast(model.yahoo_lstm_model, np.array(numbers), model.yahoo_scaler)

            fig2 = xgb_forecast(model.yahoo_xgb_model, np.array(numbers), model.yahoo_scaler)

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig1)
            with col2:
                st.pyplot(fig2)
        except ValueError:
            st.error("Please enter valid numbers separated by commas.")
    