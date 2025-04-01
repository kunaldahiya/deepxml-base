import torch
from torch import Tensor


class _Linear(torch.nn.Linear):
    """Linear layer (with minor changes)"""
    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool=True,
            device: str=None) -> None:
        """
        Args:
            input_size (int): input size of transformation
            output_size (int): output size of transformation
            bias (bool, optional): use bias term. Defaults to True.
            device (str, optional): device. Defaults to "cuda".
        """
        super(_Linear, self).__init__(input_size, output_size, bias, device)

    def get_weights(self) -> Tensor:
        """Get weights as a torch tensor
        Bias is appended in the end
        """
        _wts = self.weight.detach().cpu()
        if self.bias is not None:
            _bias = self.bias.detach().cpu().reshape(-1, 1)
            _wts = torch.hstack([_wts, _bias])
        return _wts

    def initialize(self, weight: Tensor):
        """Initialize weight and bias from existing tensors

        Args:
            weight (Tensor): weight matrix
            Last column will be considered as bias if bias is used
        """
        if self.bias:
            weight, bias = weight[:, :-1], weight[:, -1]
        self.weight.data.copy_(weight.to(self.weight.data.device))
        if self.bias is not None:
            self.bias.data.copy_(bias.to(self.bias.data.device))

    @property
    def sparse(self) -> bool:
        return False
