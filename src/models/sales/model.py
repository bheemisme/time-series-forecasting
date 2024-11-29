import torch
import torch.nn as nn
import joblib

class SalesLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SalesLSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
             nn.ReLU(),
             nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


input_size = 1
hidden_size = 10
num_layers = 1
output_size = 1

sales_lstm_model = SalesLSTMModel(input_size, hidden_size, 
                                  num_layers, output_size)

sales_lstm_model.load_state_dict(
    torch.load("./src/models/sales/sales_lstm_model.pth", weights_only=True, map_location='cpu')
)


sales_xgb_model = joblib.load('./src/models/sales/num_sales_xgb_model.joblib')
num_sales_arima = joblib.load('./src/models/sales/num_sales_model_arima_fit.joblib')
sales_model_arima = joblib.load('./src/models/sales/sales_model_fit.joblib')

