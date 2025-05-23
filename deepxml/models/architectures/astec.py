from typing import Optional, Callable

import torch.nn as nn
from torch import Tensor, LongTensor
from .embedding import Embedding


class Astec(nn.Module):
    """Encode a document using the feature representaion as per Astec paper
        (Bag-of-embedding followed by a non-linearity)


    Args:
        vocabulary_dims (int): vocalubary size
        embedding_dims (int, optional): dimension of embeddings. Defaults to 300
        dropout (float, optional): drop probability. Defaults to 0.5.
        padding_idx (int, optional): padding index. Defaults to 0.
            * index for <PAD>; embedding is not updated
            * Values other than 0 are not yet tested
        reduction (Optional[str], optional): reduction. Defaults to 'sum'.
            * None: don't reduce
            * sum: sum over tokens
            * mean: mean over tokens
        nnl (Optional[str], optional): non linearity. Defaults to 'gelu'.
            * None: linear 
            * relu / gelu : non-linearity
        sparse (bool, optional): sparse gradients? Defaults to True.
            sparse or dense gradients
            * the optimizer will infer from this parameters
        freeze (bool, optional): freeze parameters? Defaults to False.
            * freeze the gradient of token embeddings
        device (str, optional): Keep embeddings on device. Defaults to "cuda".
    """
    def __init__(
            self,
            vocabulary_dims: int,
            embedding_dims: int = 300,
            dropout: float = 0.5,
            padding_idx: int = 0,
            reduction: Optional[str] = 'sum',
            nnl: Optional[str] = 'gelu',
            sparse: bool = True,
            freeze: bool = False,
            device: str = "cuda") -> None:
        super(Astec, self).__init__()
        self.vocabulary_dims = vocabulary_dims + 1
        self.embedding_dims = embedding_dims
        self.padding_idx = padding_idx
        self.device = device
        self.sparse = sparse
        self.reduction = reduction
        self.embeddings = self._construct_embedding()
        self.nnl = self._get_nnl(nnl)
        self.dropout = nn.Dropout(dropout)
        self.freeze = freeze
        if self.freeze:
            for params in self.embeddings.parameters():
                params.requires_grad = False

    def _get_nnl(self, nnl) -> Callable:
        if nnl == 'relu':
            return nn.ReLU()
        elif nnl == 'identity' or nnl == 'null':
            return nn.Identity()
        elif nnl == 'gelu':
            return nn.GELU()
        else:
            raise NotImplementedError("Unknown non linearity")

    def _construct_embedding(self) -> Callable:
        return Embedding(
            num_embeddings=self.vocabulary_dims,
            embedding_dim=self.embedding_dims,
            padding_idx=self.padding_idx,
            scale_grad_by_freq=False,
            device=self.device,
            reduction=self.reduction,
            sparse=self.sparse)

    def encoder(self, x: Tensor, x_ind: Optional[LongTensor]) -> Tensor:
        return self.encode(x, x_ind)

    def encode(self, x: Tensor, x_ind: Optional[LongTensor]) -> Tensor:
        if x_ind is None:  # Assume embedding is pre-computed
            return x
        else:
            return self.embeddings(x_ind, x)

    def forward(self, x: tuple) -> Tensor:
        """
        Arguments:
        ----------
        x: (torch.Tensor or None, torch.LongTensor)
            token weights and indices
            weights can be None

        Returns:
        --------
        embed: torch.Tensor
            transformed document representation
            Dimension depends on reduction
        """
        return self.dropout(self.nnl(self.encoder(*x)))

    def to(self) -> None:
        super().to(self.device)

    def initialize(self, x: Tensor) -> None:
        self.embeddings.from_pretrained(x)

    def initialize_token_embeddings(self, x: Tensor) -> None:
        return self.initialize(x)

    def get_token_embeddings(self) -> Tensor:
        return self.embeddings.get_weights()

    @property
    def representation_dims(self) -> int:
        return self.embedding_dims
 
