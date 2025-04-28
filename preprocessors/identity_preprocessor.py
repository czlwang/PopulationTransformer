from torch import nn

class IdentityPreprocessor(nn.Module):
    def __init__(self, cfg):
        super(IdentityPreprocessor, self).__init__()

    def forward(self, x):
        return x
