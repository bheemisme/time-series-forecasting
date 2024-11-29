import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def lstm_predict(model: nn.Module,inputs: np.ndarray,attr: str,scaler = None):
    X = pd.DataFrame({
        attr: inputs
    })

    if scaler:
        X = scaler.transform(X)
    X = torch.tensor(X).type(torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        y = model(X)

    X = X.squeeze(2)

    if scaler is not None:
        X = scaler.inverse_transform(X.cpu().numpy())
        y = scaler.inverse_transform(y.cpu().numpy())
    
    return X[0], y[0][0]

def xgb_predict(model, inputs: np.ndarray, attr: str, scaler = None):
    X = pd.DataFrame({
        attr: inputs
    })

    if scaler:
        X = scaler.transform(X)
    X = X.reshape(1, -1)  # type: ignore
    y = model.predict(X).reshape(-1,1)
    
    if scaler is not None:
        X = scaler.inverse_transform(X)
        y = scaler.inverse_transform(y)

    return X[0], y[0][0]

def lstm_forecast(model, inputs, attr,  scaler=None):
    inputs, out = lstm_predict(model, inputs, attr, scaler)
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

def xgb_forecast(model, inputs, attr, scaler=None):
    inputs, out = xgb_predict(model, inputs, attr, scaler)
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
