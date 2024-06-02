import torch.nn as nn


class ProsodyEncoder(nn.Module):
    def __init__(self, input_size=5, hidden_size=384, num_layers=2, dropout=0.1):
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
        self.linear = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, x, output_hidden_states=True, **kwargs):
        x = self.lstm(x)[0]
        x = self.relu(x)
        x = self.linear(x)
        if output_hidden_states:
            return ModelOutput([x])
        return x


class TransformerProsodyEncoder(nn.Module):
    def __init__(self, input_size=5, hidden_size=192, num_layers=2, dropout=0.1, heads=2):
        super(TransformerProsodyEncoder, self).__init__()
        self.heads = heads
        self.projection = nn.Linear(input_size, hidden_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=heads,
                dim_feedforward=hidden_size,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, x, output_hidden_states=True, **kwargs):
        x = self.projection(x)
        x = self.encoder(x)
        if output_hidden_states:
            return ModelOutput([x])
        return x



class ModelOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states
