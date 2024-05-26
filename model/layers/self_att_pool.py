import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionPooling(nn.Module):

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, x, **kwargs):
        weights = self.W(x).squeeze(-1)
        attention_scores = F.softmax(weights).unsqueeze(-1)

        pooled_output = torch.sum(torch.mul(x, attention_scores), dim=1)
        return pooled_output
