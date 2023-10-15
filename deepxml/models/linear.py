import torch
import torch.nn as nn
from torch import Tensor, LongTensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter

__author__ = 'KD'


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
        torch.nn.init.xavier_uniform_(
            self.weight.data,
            gain=torch.nn.init.calculate_gain('relu'))
        if self.bias is not None:
            self.bias.data.fill_(0)

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


class SparseLinear(Linear):
    """Sparse Linear linear with sparse gradients

    Args:
        input_size (int): input size of transformation
        output_size (int): output size of transformation
        padding_idx (int): index for dummy label; embedding is not updated
        bias (bool, optional): use bias term. Defaults to True.
        device (str, optional): device. Defaults to "cuda".
    """
    def __init__(
            self,
            input_size: int,
            output_size: int,
            padding_idx=None,
            bias: bool=True,
            device: str="cuda") -> None:
        self.padding_idx = padding_idx
        super(SparseLinear, self).__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            device=device)

    def forward(self, embed: Tensor, shortlist: LongTensor) -> Tensor: 
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
        embed = embed.to(self.device)
        shortlist = shortlist.to(self.device)
        short_weights = F.embedding(shortlist,
                                    self.weight,
                                    sparse=self.sparse,
                                    padding_idx=self.padding_idx)
        out = torch.matmul(embed.unsqueeze(1), short_weights.permute(0, 2, 1))
        if self.bias is not None:
            short_bias = F.embedding(shortlist,
                                     self.bias,
                                     sparse=self.sparse,
                                     padding_idx=self.padding_idx)
            out = out + short_bias.permute(0, 2, 1)
        return out.squeeze()

    def reset_parameters(self) -> None:
        """Initialize weights vectors
        """
        super().reset_parameters()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def __repr__(self) -> str:
        s = '{name}({input_size}, {output_size}, {device}'
        if self.bias is not None:
            s += ', bias=True'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        s += ', sparse=True)'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def get_weights(self) -> torch:
        """Get weights as numpy array
        Bias is appended in the end
        """
        _wts = self.weight.detach().cpu()
        if self.padding_idx is not None:
            _wts = _wts[:-1, :]
        if (self.bias is not None):
            _bias = self.bias.detach().cpu()
            if self.padding_idx is not None:
                _bias = _bias[:-1, :]
            _wts = torch.hstack([_wts, _bias])
        return _wts

    @property
    def sparse(self) -> bool:
        return True


class SharedSparseLinear(Linear):
    """Sparse Linear layer with sparse gradients
        * the shortlist is shared across the documents
        * useful in case of implicit sampling or in-batch sampling

    Args:
        input_size (int): input size of transformation
        output_size (int): output size of transformation
        padding_idx (int): index for dummy label; embedding is not updated
        bias (bool, optional): use bias term. Defaults to True.
        device (str, optional): device. Defaults to "cuda".
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            padding_idx=None,
            bias: bool=True,
            device: str="cuda") -> None:
        self.padding_idx = padding_idx
        super(SharedSparseLinear, self).__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            device=device)

    def forward(self, embed: Tensor, shortlist: LongTensor) -> Tensor:
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
        embed = embed.to(self.device)
        shortlist = shortlist.to(self.device)
        short_weights = F.embedding(shortlist,
                                    self.weight,
                                    sparse=self.sparse,
                                    padding_idx=self.padding_idx)
        out = embed @ short_weights.T
        if self.bias is not None:
            short_bias = F.embedding(shortlist,
                                     self.bias,
                                     sparse=self.sparse,
                                     padding_idx=self.padding_idx)
            out = out + short_bias.view(1, -1)
        return out

    def reset_parameters(self) -> None:
        """Initialize weights vectors
        """
        super().reset_parameters()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def __repr__(self) -> str:
        s = '{name}({input_size}, {output_size}, {device}'
        if self.bias is not None:
            s += ', bias=True'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        s += ', sparse=True)'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def get_weights(self) -> torch:
        """Get weights as numpy array
        Bias is appended in the end
        """
        _wts = self.weight.detach().cpu()
        if self.padding_idx is not None:
            _wts = _wts[:-1, :]
        if (self.bias is not None):
            _bias = self.bias.detach().cpu()
            if self.padding_idx is not None:
                _bias = _bias[:-1, :]
            _wts = torch.hstack([_wts, _bias])
        return _wts

    @property
    def sparse(self) -> bool:
        return True


class UNSparseLinear(SparseLinear):
    """Sparse Linear layer with sparse gradients
       * will normalize document and label representations to unit l2 norm
       * TODO: bias is ignored

    Args:
        input_size (int): input size of transformation
        output_size (int): output size of transformation
        padding_idx (int): index for dummy label; embedding is not updated
        bias (bool, optional): use bias term. Defaults to True.
        device (str, optional): device. Defaults to "cuda".
    """
    def __init__(
            self,
            input_size: int,
            output_size: int,
            padding_idx=None,
            bias: bool=True,
            device: str="cuda") -> None:
        super(UNSparseLinear, self).__init__(
            input_size=input_size,
            output_size=output_size,
            padding_idx=padding_idx,
            bias=False,
            device=device)

    def forward(self, embed: Tensor, shortlist: LongTensor) -> Tensor: 
        """Forward pass with input feature and per label shortlist
            * sparse gradients
            * assumes a per-document shortlist of labels is available
            * unit normalized embeddings and shortlist
            
        Args:
            input (Tensor): input tensor
                shape (batch size, input size)
            shortlist (LongTensor): shortlist for each data point
                shape (batch size, shortlist size)
            
        Returns:
            Tensor: score for each label in shortlist for each document
                shape (batch size, shortlist size)
        """
        embed = F.normalize(embed.to(self.device), dim=1)
        shortlist = shortlist.to(self.device)
        short_weights = F.embedding(shortlist,
                                    self.weight,
                                    sparse=self.sparse,
                                    padding_idx=self.padding_idx)
        short_weights = F.normalize(short_weights, dim=2)
        out = torch.matmul(embed.unsqueeze(1), short_weights.permute(0, 2, 1))
        return out.squeeze()
 

class UNSSparseLinear(SparseLinear):
    """Sparse Linear layer with sparse gradients
        * the shortlist is shared across the documents
        * useful in case of implicit sampling or in-batch sampling
        * will normalize document and label representations to unit l2 norm
        * TODO: bias is ignored

    Args:
        input_size (int): input size of transformation
        output_size (int): output size of transformation
        padding_idx (int): index for dummy label; embedding is not updated
        bias (bool, optional): use bias term. Defaults to True.
        device (str, optional): device. Defaults to "cuda".
    """
    def __init__(
            self,
            input_size: int,
            output_size: int,
            padding_idx=None,
            bias: bool=True,
            device: str="cuda") -> None:
        super(UNSSparseLinear, self).__init__(
            input_size=input_size,
            output_size=output_size,
            padding_idx=padding_idx,
            bias=False,
            device=device)

    def forward(self, embed: Tensor, shortlist: LongTensor) -> Tensor:
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
        shortlist = shortlist.to(self.device)
        short_weights = F.embedding(shortlist,
                                    self.weight,
                                    sparse=self.sparse,
                                    padding_idx=self.padding_idx)
        out = F.normalize(embed, dim=-1) @ F.normalize(short_weights, dim=-1).T
        return out
 
    @property
    def sparse(self) -> bool:
        return True
