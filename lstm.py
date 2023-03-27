import torch
from torch import nn

import config


class LSTM(nn.Module):
    def __init__(
            self,
            input_size=config.LSTM_INPUT_SIZE,
            hidden_size=config.LSTM_HIDDEN_SIZES,
            num_layers=config.LSTM_NUMS_LAYERS,
            num_classes=config.LSTM_NUM_CLASSES,
            device='cpu'
    ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[0])
        return out
