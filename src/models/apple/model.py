import torch.nn as nn
import torch
import joblib

class AppleLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(AppleLSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            dropout=dropout)
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

apple_lstm_model = AppleLSTMModel(input_size, hidden_size,
                                  num_layers, output_size,
                                  dropout)

apple_lstm_model.load_state_dict(torch.load('./src/models/apple/apple_lstm_model.pth',
                                            map_location= 'cpu', weights_only=False))


apple_scaler = joblib.load('./src/models/apple/apple_close_scaler.joblib')
apple_xgb_model = joblib.load('./src/models/apple/apple_xgb_model.joblib')

