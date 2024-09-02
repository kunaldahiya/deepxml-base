import math

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Linear(nn.Module):
    """Linear layer

    Args:
        input_size (int): input size of transformation
        output_size (int): output size of transformation
        bias (bool, optional): use bias term. Defaults to True.
        device (str, optional): device. Defaults to "cuda".
    """
    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool=True,
            device: str="cuda") -> None:
        super(Linear, self).__init__()
        self.device = device  # Useful in case of multiple GPUs
        self.input_size = input_size
        self.output_size = output_size
        self.weight = Parameter(torch.Tensor(self.output_size, self.input_size))
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass

        Args:
            input (Tensor): input tensor

        Returns:
            Tensor: transformed tensor
        """
        if self.bias is not None:
            return F.linear(input.to(self.device), self.weight, self.bias.view(-1))
        else:
            return F.linear(input.to(self.device), self.weight)

    def to(self) -> None:
        """Transfer to device
        """
        super().to(self.device)

    def reset_parameters(self) -> None:
        """Initialize vectors
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def get_weights(self) -> Tensor:
        """Get weights as a torch tensor
        Bias is appended in the end
        """
        _wts = self.weight.detach().cpu()
        if self.bias is not None:
            _bias = self.bias.detach().cpu()
            _wts = torch.hstack([_wts, _bias])
        return _wts

    def __repr__(self) -> str:
        s = '{name}({input_size}, {output_size}, {device}'
        if self.bias is not None:
            s += ', bias=True'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    @property
    def sparse(self) -> bool:
        return False
