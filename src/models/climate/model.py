import torch
import torch.nn as nn
import joblib
import kagglehub
import os

path = kagglehub.model_download("sudarshan1927/time-series-forecasting/other/climate")

class ClimateModel(nn.Module):

  def __init__(self, input_size, hidden_size, output_size, num_layers):
    super(ClimateModel, self).__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.linear = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    lstm_out, _ = self.lstm(x)
    out = self.linear(lstm_out[:, -1, :])
    return out
  
input_size = 1
hidden_size = 8
num_layers = 1
output_size= 1

climate_model = ClimateModel(input_size=input_size,
                  hidden_size=hidden_size,
                  output_size = output_size,
                  num_layers=num_layers)



climate_model.load_state_dict(
    torch.load(os.path.join(path, 'climate_model.pth'), 
               weights_only=True, map_location='cpu')
)


climate_xgb_model = joblib.load(os.path.join(path, 'climate_xgb_model.joblib'))
climate_scaler = joblib.load(os.path.join(path, 'climate_scaler.joblib'))



