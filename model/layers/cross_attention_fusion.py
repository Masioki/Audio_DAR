import torch
import torch.nn as nn

from model.layers.cross_attention import CrossAttention
from model.layers.lambd import LambdaLayer


class CrossAttentionFusion(nn.Module):
    def __init__(self,
                 q_size,
                 k1_size,
                 k2_size,
                 heads=8,
                 fusion_strategy='dense',
                 ):
        super(CrossAttentionFusion, self).__init__()

        self.c1 = CrossAttention(q_size, k1_size, heads)
        self.c2 = CrossAttention(q_size, k2_size, heads)

        if fusion_strategy == 'dense':
            self.linear = nn.Linear(2 * q_size, q_size)
            self.fusion = LambdaLayer(lambda c1_out, c2_out: self.linear(torch.cat([c1_out, c2_out], dim=-1)))
        elif fusion_strategy == 'mean':
            self.fusion = LambdaLayer(lambda c1_out, c2_out: torch.mean(torch.stack([c1_out, c2_out], dim=-1), dim=-1))

    def forward(self, q, k1, k2, k1_mask=None, k2_mask=None):
        c1_out = self.c1(q, k1, k1_mask)
        c2_out = self.c2(q, k2, k2_mask)
        return self.fusion(c1_out, c2_out)
