from typing import Optional, Callable, Any

import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class Embedding(nn.Module):
    """General way to handle embeddings

    * Support for sequential models
    * Memory efficient way to compute weighted EmbeddingBag

    Args:
        vocabulary_dims (int): vocalubary size        
        embedding_dims (int, optional): dimension of embeddings. Defaults to 300
        padding_idx (int, optional): padding index. Defaults to 0.
            * index for <PAD>; embedding is not updated
            * Values other than 0 are not yet tested
        max_norm (Optional[float], optional): control norm. Defaults to None.
            Constrain norm to this value 
        norm_type (int, optional): norm type in max_norm. Defaults to 2.
        scale_grad_by_freq (bool, optional): scale gradients. Defaults to False.
        sparse (bool, optional): sparse gradients? Defaults to True.
            sparse or dense gradients
            * the optimizer will infer from this parameters
        reduction (bool, optional): reduction function. Defaults to True.
            * None: don't reduce
            * sum: sum over tokens
            * mean: mean over tokens
        pretrained_weights (bool, optional): _description_. Defaults to None.
            Initialize with these weights
            * first token is treated as a padding index
            * dim=1 should be one less than the num_embeddings
        device (str, optional): Keep embeddings on device. Defaults to "cuda".
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: Optional[str] = None,
            max_norm: Optional[float] = None,
            norm_type: int = 2,
            scale_grad_by_freq: bool = False,
            sparse: bool = False,
            reduction: bool = True,
            pretrained_weights: bool = None,
            device: str ="cuda") -> None:
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.weight = Parameter(Tensor(num_embeddings, embedding_dim))
        self.sparse = sparse
        self.reduce = self._construct_reduce(reduction)
        self.reduction = reduction
        self.device = torch.device(device)
        self.reset_parameters()
        if pretrained_weights is not None:
            self.from_pretrained(pretrained_weights)

    def _construct_reduce(self, reduction: str) -> Callable:
        if reduction is None:
            return self._reduce
        elif reduction == 'sum':
            return self._reduce_sum
        elif reduction == 'mean':
            return self._reduce_mean
        else:
            return NotImplementedError(f"Unknown reduction: {reduction}")

    def reset_parameters(self) -> None:
        """
            Reset weights
        """
        nn.init.xavier_uniform_(
            self.weight.data, gain=nn.init.calculate_gain('relu'))
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def to(self) -> None:
        super().to(self.device)

    def _reduce_sum(self, x: Tensor, w: Optional[Tensor]) -> Tensor:
        if w is None:
            return torch.sum(x, dim=1)
        else:
            return torch.sum(x * w.unsqueeze(2), dim=1)

    def _reduce_mean(self, x: Tensor, w: Optional[Tensor]) -> Tensor:
        if w is None:
            return torch.mean(x, dim=1)
        else:
            return torch.mean(x * w.unsqueeze(2), dim=1)

    def _reduce(self, x: Tensor, *args: Optional[Any]) -> Tensor:
        return x

    def forward(self, x: LongTensor, w: Optional[Tensor] = None) -> Tensor:
        """Forward pass for embedding layer

        Args:
            x (LongTensor): indices of tokens in a batch
                shape: (batch_size, max_features_in_a_batch)
            w (Optional[Tensor], optional): weights of tokens. Defaults to None.
                shape: (batch_size, max_features_in_a_batch)

        Returns:
            Tensor: embedding for each sample
                Shape: (batch_size, seq_len, embedding_dims), if reduction is None
                Shape: (batch_size, embedding_dims), otherwise

        """
        x = F.embedding(
            x, self.weight,
            self.padding_idx, self.max_norm, self.norm_type,
            self.scale_grad_by_freq, self.sparse)
        return self.reduce(x, w)

    def from_pretrained(self, embeddings: Tensor) -> None:
        # first index is treated as padding index
        assert embeddings.shape[0] == self.num_embeddings-1, \
            "Shapes doesn't match for pre-trained embeddings"
        self.weight.data[1:, :] = embeddings

    def get_weights(self) -> Tensor:
        return self.weight.detach().cpu()[1:, :]

    def __repr__(self) -> str:
        s = '{name}({num_embeddings}, {embedding_dim}, {device}'
        s += ', reduction={reduction}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
 
