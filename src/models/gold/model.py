import torch.nn as nn
import torch
import joblib
import kagglehub
import os


path = kagglehub.model_download("sudarshan1927/time-series-forecasting/other/gold")

class GoldLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(GoldLSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

input_size = 1 
hidden_size = 10
num_layers = 1
output_size = 1
learning_rate = 0.001
dropout = 0

gold_lstm_model = GoldLSTMModel(input_size, hidden_size,
                            num_layers, output_size, dropout)

gold_lstm_model.load_state_dict(torch.load(os.path.join(path, 'gold_lstm_model.pth'),
                                            map_location= 'cpu', weights_only=False))

gold_scaler = joblib.load(os.path.join(path, 'gold_scaler.joblib'))

gold_xgb_model = joblib.load(os.path.join(path, 'gold_xgb_model.joblib'))
