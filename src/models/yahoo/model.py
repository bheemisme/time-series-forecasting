import torch
import torch.nn as nn
import joblib

class YahooLSTMModel(nn.Module):
    def __init__(self, input_size,
                 hidden_size,
                 num_layers,
                 output_size):
        super(YahooLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)

        self.fc = nn.Sequential(
              nn.ReLU(),
              nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


input_size = 1
hidden_size = 128
num_layers = 1
output_size = 1

yahoo_lstm_model = YahooLSTMModel(input_size, hidden_size, num_layers, output_size)
yahoo_lstm_model.load_state_dict(
    torch.load("./src/models/yahoo/yahoo_lstm_model.pth", weights_only=True, map_location='cpu')
)

yahoo_scaler = joblib.load('./src/models/yahoo/yahoo_scaler.joblib')
yahoo_xgb_model = joblib.load('./src/models/yahoo/yahoo_xgb_model.joblib')

