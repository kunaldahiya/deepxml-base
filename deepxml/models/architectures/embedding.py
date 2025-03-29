from typing import Optional, Callable, Any
from numpy import ndarray

import math
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor, LongTensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from xclib.utils.sparse import normalize
from xclib.utils.clustering import cluster_balance, b_kmeans_dense


class Embedding(nn.Module):
    """General way to handle embeddings

    * Support for sequential models
    * Memory efficient way to compute weighted EmbeddingBag

    Args:
        vocabulary_dims (int): vocalubary size        
        embedding_dim (int, optional): dimension of embeddings. Defaults to 300
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
                Shape: (batch_size, seq_len, embedding_dim), if reduction is None
                Shape: (batch_size, embedding_dim), otherwise

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
 

class EmbeddingBank(torch.nn.Module):
    """Embedding bank to store some vectors
        - supports mapping (can be one-to-one or many-to-one)
        - supports clustering
    """
    def __init__(
            self, 
            embedding_dim: int, 
            num_embeddings: int,
            num_items: int=None, 
            mapping: ndarray=None, 
            requires_grad: bool=True,
            device: str="cpu") -> None:
        """
        Args:
            embedding_dim (int): Embedding dimensionality
            num_embeddings (int): Store these many embeddings
            num_items (int, optional): Defaults to None.
                Total number of items (as per original data)
                - one-to-one: each item will have its separate embedding
                - many-to-one: multiple items may share embeddings
            mapping (np.ndarray, optional): Defaults to None.
                item to embedding mapping.
                - if mapping is None and num_items == num_embeddings
                    then simple arange is used as mapping
            requires_grad (bool, optional): Want to update the bank? Defaults to True.
            device (str, optional): _description_. Defaults to "cpu".
        """
        super(EmbeddingBank, self).__init__()

        if mapping is not None and num_items is None:
            num_items = len(mapping)

        self.padding_idx = 0
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.num_items = num_items
        self.weight = torch.nn.Parameter(
            torch.Tensor(num_embeddings, embedding_dim),
            requires_grad=requires_grad)
        self.device = device

        if mapping is None and num_items == num_embeddings:
            mapping = np.arange(num_items)

        if mapping is not None:
            self.register_buffer('mapping', torch.LongTensor(mapping))
        else:
            self.register_buffer('mapping', torch.zeros((num_items,)).long())
        self.initialize()

    def _power_two_check(self, n: int) -> bool:
        assert math.log2(n).is_integer(), \
            "n_embeddings must be power of 2 (with current clustering style)"

    def _cluster_and_set_mapping(
            self, 
            X: ndarray, 
            num_threads: int=6) -> None:
        """Cluster and set mapping

        Args:
            X (ndarray): Representations of items
            num_threads (int, optional): #threads to use. Defaults to 6.
        """
        self._power_two_check(self.num_embeddings)
        _, mapping = cluster_balance(
            X=normalize(X.astype('float32'), copy=True),
            clusters=[np.arange(len(X), dtype='int')],
            num_clusters=self.num_embeddings,
            splitter=b_kmeans_dense,
            num_threads=num_threads,
            verbose=True)
        self.set_mapping(mapping)

    def cluster_and_set_mapping(
            self, 
            X: ndarray, 
            num_threads: int=6) -> None:
        """Cluster and set mapping

        Args:
            X (ndarray): Representations of items
            num_threads (int, optional): #threads to use. Defaults to 6.
        """
        self._cluster_and_set_mapping(X, num_threads)

    def set_mapping(self, mapping: ndarray) -> None:
        """Manually set the mapping when available

        Args:
            mapping (ndarray): mapping from item to vector in bank
        """
        self.mapping.copy_(
            torch.LongTensor(mapping).to(self.mapping.get_device()))

    def __getitem__(self, ind: int | Tensor) -> Tensor:
        return self.weight[self.mapping[ind]].squeeze()
    
    def forward(self, ind: int | Tensor) -> Tensor:
        return self.__getitem__(ind)

    @property
    def repr_dims(self) -> int:
        return self.embedding_dim

    def initialize(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight.data)

    def __repr__(self) -> str:
        s = '{name}({num_embeddings}, {embedding_dim}, {num_items}, {device})'
        return s.format(name=self.__class__.__name__, **self.__dict__)
