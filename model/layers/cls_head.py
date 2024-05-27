import torch.nn as nn


class SentenceClassifierHead(nn.Module):
    def __init__(self, features, hid_size, labels, dropout=0.3):
        super(SentenceClassifierHead, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(features, hid_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hid_size, labels),
        )

    def forward(self, x):
        return self.model(x)
