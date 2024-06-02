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
        self.layer_norm = nn.LayerNorm(q_size)
        if fusion_strategy == 'dense':
            self.linear = nn.Linear(2 * q_size, q_size)
            self.relu = nn.ReLU()
            self.fusion = LambdaLayer(
                lambda c1_out, c2_out: self.relu(self.linear(torch.cat([c1_out, c2_out], dim=-1))))
        elif fusion_strategy == 'mean':
            self.fusion = LambdaLayer(lambda c1_out, c2_out: torch.mean(torch.stack([c1_out, c2_out], dim=-1), dim=-1))

    def forward(self, q, k1, k2, k1_mask=None, k2_mask=None):
        c1_out = self.layer_norm(self.c1(q, k1, k1_mask))
        c2_out = self.layer_norm(self.c2(q, k2, k2_mask))
        return self.fusion(c1_out, c2_out)


class CrossAttentionFusionModule(nn.Module):
    def __init__(self, q_size,
                 k1_size,
                 k2_size,
                 heads=8,
                 fusion_strategy='dense',
                 layers: int = 2):
        super(CrossAttentionFusionModule, self).__init__()
        self.layers = nn.ModuleList([
            CrossAttentionFusion(q_size, k1_size, k2_size, heads, fusion_strategy)
            for _ in range(layers)
        ])

    def forward(self, q, k1, k2, k1_mask=None, k2_mask=None):
        for layer in self.layers:
            q = layer(q, k1, k2, k1_mask, k2_mask)
        return q
