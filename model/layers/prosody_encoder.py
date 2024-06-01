import torch.nn as nn


class ProsodyEncoder(nn.Module):
    def __init__(self, input_size=5, hidden_size=10, num_layers=2, dropout=0.1, out_size=384):
        super(ProsodyEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.relu = nn.ReLU()
        self.linear = nn.Linear(2 * hidden_size, out_size)

    def forward(self, x, output_hidden_states=True, **kwargs):
        x = self.lstm(x)[0]
        x = self.relu(x)
        x = self.linear(x)
        if output_hidden_states:
            return ModelOutput([x])
        return x


class ModelOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states
