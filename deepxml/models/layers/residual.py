from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    """Implementation of a Residual block with constrained spectral norm

    Args:
        input_size (int): input dimension
        output_size (int): output dimension
        dropout (float): dropout probability
        init (str, optional): initialization strategy. Defaults to 'eye'.
    """
    def __init__(
            self,
            input_size: int,
            output_size: int,
            dropout: float,
            init: str='eye') -> None:
        super(Residual, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.init = init
        self.dropout = dropout
        self.padding_size = self.output_size - self.input_size
        self.hidden_layer = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.input_size, self.output_size)),
            nn.ReLU(),
            nn.Dropout(self.dropout))
        self.initialize(self.init)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass for Residual

        Args:
            input (Tensor): dense document embedding

        Returns:
            Tensor: document embeddings transformed via residual block
        """
        temp = F.pad(input, (0, self.padding_size), 'constant', 0)
        input = self.hidden_layer(input) + temp
        return input

    def initialize(self, init_type: str) -> None:
        """Initialize units

        Args:
            init_type (str): Initialize hidden layer
              * 'random' or 'eye'
        """
        if init_type == 'random':
            nn.init.xavier_uniform_(
                self.hidden_layer[0].weight,
                gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(self.hidden_layer[0].bias, 0.0)
        else:
            nn.init.eye_(self.hidden_layer[0].weight)
            nn.init.constant_(self.hidden_layer[0].bias, 0.0)