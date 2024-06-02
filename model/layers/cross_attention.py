import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):

    def __init__(self, query_dim, key_dim, heads: int = 8):
        super(CrossAttention, self).__init__()
        self.heads = heads
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = key_dim
        self.head_dim = key_dim // heads

        assert key_dim % heads == 0, "Key value dim must be divisible by number of heads"

        self.query_projection = nn.Linear(self.query_dim, self.key_dim)
        self.key_projection = nn.Linear(self.key_dim, self.key_dim)
        self.value_projection = nn.Linear(self.value_dim, self.key_dim)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

        self.output_projection = nn.Linear(self.key_dim, self.query_dim)

    def forward(self, q, k, mask=None):
        query = self.query_projection(q)
        key = self.key_projection(k)
        value = self.value_projection(k)

        query = query.view(query.size(0), -1, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(key.size(0), -1, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(value.size(0), -1, self.heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale.to(query.device)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e10)

        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.transpose(1, 2).contiguous().view(query.size(0), -1,
                                                                              self.heads * self.head_dim)

        return self.output_projection(attention_output)
