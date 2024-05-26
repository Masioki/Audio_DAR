import torch.nn as nn


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, *args, **kwargs):
        return self.lambd(*args, **kwargs)
