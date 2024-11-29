import streamlit as st
import numpy as np
from src.models.sales import model
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def lstm_predict(model: nn.Module,inputs: np.ndarray):
    inputs = np.log(inputs)
    X = torch.tensor(inputs).type(torch.float32).unsqueeze(0)
    X = X.unsqueeze(2)
    model.eval()
    with torch.no_grad():
        y = model(X)

    X = X.squeeze(2)
    X = np.exp(X.cpu().numpy())
    y = np.exp(y.cpu().numpy())
    return X[0], y[0][0]

def xgb_predict(model, inputs: np.ndarray):
    inputs = np.log(inputs)
    X = torch.tensor(inputs).type(torch.float32).unsqueeze(0)

    
    X = X.reshape(1, -1)  # type: ignore
    y = model.predict(X).reshape(-1,1)
    
    X = np.exp(X)
    y = np.exp(y)
    
    return X[0], y[0][0]

def lstm_forecast(model, inputs):
    inputs, out = lstm_predict(model, inputs)
    fig = plt.figure()
    plt.plot(np.arange(1,len(inputs)+1),inputs, 
             linestyle='dotted', label='input', 
             marker='o', markersize=5)
    plt.plot(np.arange(len(inputs), len(inputs)+2),np.array([inputs[-1],out]), 
             linestyle='dotted', label='forecast',
             color='green', marker='o', markersize=5)
    plt.legend()
    plt.title(f'LSTM Forecasting, forecast={out}')
    return fig

def xgb_forecast(model, inputs):
    inputs, out = xgb_predict(model, inputs)
    fig = plt.figure()
    plt.plot(np.arange(1,len(inputs)+1),inputs, 
             linestyle='dotted', label='input',  
             marker='o', markersize=5)
    plt.plot(np.arange(len(inputs), len(inputs)+2),np.array([inputs[-1],out]), 
             linestyle='dotted', label='forecast',
             color='green', marker='o', markersize=5)
    plt.legend()
    plt.title(f'XGB Forecasting, forecast = {out}')
    return fig

def arima_forecast(model, last: list, steps: int, title: str):
    out = model.forecast(steps=steps)
    out = np.exp(out).to_numpy()
    # last = np.ndarray(last, dtype=np.float32)
    last.append(out[0])

    # print(last, out)
    fig = plt.figure()
    plt.plot(np.arange(1, len(last)+1), last, linestyle='dotted',
             label='previous', marker='o', markersize=5)
    plt.plot(np.arange(len(last), len(last)+len(out)), out, linestyle='dotted',
              label='forecast', marker='o', markersize=5)
    plt.legend()
    plt.title(title)
    return fig
def app():
    st.title('Time Series Forecasting')
    st.markdown('## Sales Forecasting')

    number_input = st.number_input(label="Enter number of days", min_value=1, max_value=30)
    if number_input:
        
        fig1 = arima_forecast(model.num_sales_arima, 
                              last=[17., 15.,  1., 17., 24., 13., 16., 13., 12.,  2.], 
                              steps=number_input, 
                              title="Number of sales")
        fig2 = arima_forecast(model.sales_model_arima,
                              last=[115.02226667,91.58707692,35.712,57.459,
                                    115.30245,255.18790909,225.38371429,101.21127273,
                                    484.5984,245.944],
                              steps=number_input,
                              title="Sales")
        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(fig1)
        
        with col2:
            st.pyplot(fig2)

        

    user_input = st.text_input("Enter number of sales for 10 days: ")

    if user_input:
        try:
            numbers = [float(num.strip()) for num in user_input.split(",")]
            
            
            numbers = np.array(numbers)
            fig1 = lstm_forecast(model.sales_lstm_model, numbers)
            fig2 = xgb_forecast(model.sales_xgb_model,numbers)

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig1)
            with col2:
                st.pyplot(fig2)
        except ValueError:
            st.error("Please enter valid numbers separated by commas.")
    