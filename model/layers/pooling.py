import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.lambd import LambdaLayer
from model.utils import wmean_pooling, mean_pooling, cls_token


class ConfigurablePooling(nn.Module):
    def __init__(self, features=None, embedding_strategy='wmean-pooling'):
        super(ConfigurablePooling, self).__init__()
        if embedding_strategy == 'wmean-pooling':
            self.model = wmean_pooling
        elif embedding_strategy == 'mean-pooling':
            self.model = mean_pooling
        elif embedding_strategy == 'cls':
            self.model = cls_token
        elif embedding_strategy == 'provided':
            self.model = lambda outputs, input_ids, attention_mask, **kwargs: outputs
        elif embedding_strategy == 'self-att':
            self.model = SelfAttentionPooling(features)
        else:
            raise Exception(f"Unknown embedding strategy: {embedding_strategy}")

        if embedding_strategy != 'self-att':
            self.model = LambdaLayer(self.model)

    def forward(self, x, input_ids=None, attention_mask=None, **kwargs):
        return self.model(x, input_ids, attention_mask, **kwargs)


class SelfAttentionPooling(nn.Module):

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, x, **kwargs):
        weights = self.W(x).squeeze(-1)
        attention_scores = F.softmax(weights).unsqueeze(-1)

        pooled_output = torch.sum(torch.mul(x, attention_scores), dim=1)
        return pooled_output
