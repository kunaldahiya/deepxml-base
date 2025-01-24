from typing import Optional, Callable

from ._linear import _Linear
from ..utils import (
    ip_sim_3d,
    cosine_sim_3d,
    ip_sim,
    cosine_sim)
from torch import Tensor, LongTensor
import torch.nn.functional as F


class OVA(_Linear): 
    """A brute-force classifier (all labels are considered for each datapoint)

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
            bias: bool = True,
            metric: str = "ip",
            device: str = None) -> None:
        super(OVA, self).__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            device=device)
        self.metric = metric
        self.similarity = self.construct_sim_func()

    def construct_sim_func(self):
        if self.metric == 'ip':
            return ip_sim
        elif self.metric == 'cosine':
            return cosine_sim
        else:
            raise NotImplementedError("Choose ip/cosine as metric")

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass

        Args:
            input (Tensor): input tensor

        Returns:
            Tensor: transformed tensor
        """
        return self.similarity(input, self.weight, self.bias)


class OVS(_Linear): 
    """A 1-vs-some classifier (each point will be provided a shortlist)

    Args:
        input_size (int): input size of transformation
        output_size (int): output size of transformation
        bias (bool, optional): use bias term. Defaults to True.
        device (str or None, optional): device. Defaults to None.
    """
    def __init__(
            self,
            input_size: int,
            output_size: int,
            padding_idx: Optional[int] = None,
            bias: bool = True,
            metric: str = "ip",
            device: str = None) -> None:
        self.padding_idx = padding_idx
        self.metric = metric
        super(OVA, self).__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            device=device)
        self.similarity = self.construct_sim_func()

    def construct_sim_func(self) -> Callable:
        if self.metric == 'ip':
            return ip_sim_3d
        elif self.metric == 'cosine':
            return cosine_sim_3d
        else:
            raise NotImplementedError("Choose ip/cosine as metric")

    def forward(self, input: Tensor, shortlist: LongTensor) -> Tensor: 
        """Forward pass with input feature and per label shortlist
            * sparse gradients
            * assumes a per-document shortlist of labels is available

        Args:
            input (Tensor): input tensor
                shape (batch size, input size)
            shortlist (LongTensor): shortlist for each data point
                shape (batch size, shortlist size)
            
        Returns:
            Tensor: score for each label in shortlist for each document
                shape (batch size, shortlist size)
        """
        _weights = F.embedding(shortlist,
                                    self.weight,
                                    sparse=self.sparse,
                                    padding_idx=self.padding_idx)
        if self.bias is not None:
            _bias = F.embedding(shortlist,
                                     self.bias,
                                     sparse=self.sparse,
                                     padding_idx=self.padding_idx)
        return self.similarity(input, _weights, _bias)

    def reset_parameters(self) -> None:
        """Initialize weights vectors
        """
        super().reset_parameters()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def get_weights(self) -> Tensor:
        """Get weights as numpy array
        Bias is appended in the end
        """
        _wts = super().get_weights()
        return _wts[:-1, :]

    @property
    def sparse(self) -> bool:
        return True


class OVSS(_Linear): 
    """A 1-vs-some classifier (each point will share a common shortlist)

    Args:
        input_size (int): input size of transformation
        output_size (int): output size of transformation
        bias (bool, optional): use bias term. Defaults to True.
        device (str or None, optional): device. Defaults to None.
    """
    def __init__(
            self,
            input_size: int,
            output_size: int,
            padding_idx: Optional[int] = None,
            bias: bool = True,
            metric: str = "ip",
            device: str = None) -> None:
        self.padding_idx = padding_idx
        self.metric = metric
        super(OVSS, self).__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            device=device)
        self.similarity = self.construct_sim_func()

    def construct_sim_func(self) -> Callable:
        if self.metric == 'ip':
            return ip_sim
        elif self.metric == 'cosine':
            return cosine_sim
        else:
            raise NotImplementedError("Choose ip/cosine as metric")

    def forward(self, input: Tensor, shortlist: LongTensor) -> Tensor: 
        """Forward pass with input feature and per label shortlist
            * sparse gradients
            * assumes a shared shortlist is available for 
              all documents in mini-batch

        Args:
            input (Tensor): input tensor
                shape (batch size, input_size)
            shortlist (LongTensor): shortlist (label indices)
                shape (shortlist size)
            
        Returns:
            Tensor: score for each document label pair
                shape (batch size, shortlist size)
        """
        _weights = F.embedding(shortlist,
                                    self.weight,
                                    sparse=self.sparse,
                                    padding_idx=self.padding_idx)
        if self.bias is not None:
            _bias = F.embedding(shortlist,
                                     self.bias,
                                     sparse=self.sparse,
                                     padding_idx=self.padding_idx)
        return self.similarity(input, _weights, _bias)

    def reset_parameters(self) -> None:
        """Initialize weights vectors
        """
        super().reset_parameters()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def get_weights(self) -> Tensor:
        """Get weights as numpy array
        Bias is appended in the end
        """
        _wts = super().get_weights()
        return _wts[:-1, :]

    @property
    def sparse(self) -> bool:
        return True
