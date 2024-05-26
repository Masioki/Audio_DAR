import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, heads: int = 8):
        super(CrossAttention, self).__init__()
        self.query_dim = query_dim // heads
        self.key_dim = key_dim // heads
        self.value_dim = self.key_dim
        self.heads = heads

        self.query = nn.Linear(query_dim, self.query_dim * heads)
        self.key = nn.Linear(key_dim, self.key_dim * heads)
        self.value = nn.Linear(key_dim, self.value_dim * heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k):
        batch_size = q.size(0)

        query = self.query(q).view(batch_size, -1, self.heads, self.query_dim).transpose(1, 2)
        key = self.key(k).view(batch_size, -1, self.heads, self.key_dim).transpose(1, 2)
        value = self.value(k).view(batch_size, -1, self.heads, self.value_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.key_dim ** 0.5)
        attention = self.softmax(scores)

        context = torch.matmul(attention, value).transpose(1, 2).contiguous().view(batch_size, -1,
                                                                                   self.heads * self.value_dim)
        return context
