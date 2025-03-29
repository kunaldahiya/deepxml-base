import torch
from .classifiers.ova import OVA, OVSS
from .layers.residual import Residual
from .architectures.astec import Astec
from .architectures.transformer import STransformer, TransformerEncoderBag


class _Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(_Identity, self).__init__()

    def forward(self, x: tuple) -> torch.Tensor:
        # useful when x_ind is None
        x, x_ind = x
        return x

    def initialize(self, *args, **kwargs):
        pass


MODS = {
    'dropout': torch.nn.Dropout,
    'batchnorm1d': torch.nn.BatchNorm1d,
    'linear': torch.nn.Linear,
    'relu': torch.nn.ReLU,
    'gelu': torch.nn.GELU,
    'residual': Residual,
    '_residual': Residual,
    'identity': torch.nn.Identity,
    '_identity': _Identity,
    'astec': Astec,
    'stransformer': STransformer,
    'transformerbag': TransformerEncoderBag,
    'ova': OVA,
    'ovss': OVSS
}
