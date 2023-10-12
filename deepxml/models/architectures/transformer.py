from typing import Callable, Union, Tuple, NoReturn

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import PreTrainedModel


class BaseTransformer(torch.nn.Module):
    """Base class for Transformers

    Args:
        transformer (Union[str, SentenceTransformer, PreTrainedModel]): from 
            Sentence Transformer or Hugginface (not tested with others)

        pooler (Union[str, None]): method to reduce the output of transformer
            * Sentence transformer uses the default pooler as per pretrained model 
            * None: return the output layer as it is
            * other pooling with HF: mean, cls, concact

        normalize (bool): return normalized outputs or as it is
    """
    def __init__(
            self,
            transformer: Union[str, SentenceTransformer, PreTrainedModel],
            pooler: Union[str, None],
            normalize: bool,
            **kwargs: dict):
        super(BaseTransformer, self).__init__()
        self.transform, self.pooler, self.normalize = self.construct(
            transformer, pooler, normalize, **kwargs)
        self.__normalize = normalize
        self.__pooler = pooler

    def construct(
            self, 
            transformer: Union[str, SentenceTransformer, PreTrainedModel],
            pooler: Union[str, None],
            normalize: bool,
            **kwargs) -> NoReturn:
        """
        Construct the transformer and the pooler
        """
        return self.construct_transformer(transformer, **kwargs), \
            self.construct_pooler(pooler, **kwargs), \
                self.construct_normalizer(normalize)

    def encode(self, x: Tuple[Tensor, Tensor]) -> Tensor:
        """Encode text based on input ids and attention mask

        Args:
            x (Tuple[Tensor, Tensor]): input ids and attention mask

        Returns:
            Tensor: encoded text
        """
        ids, mask = x
        return self.normalize(self.pooler(self.transform(ids, mask), mask))

    def forward(self, x) -> Tensor:
        """Forward pass to encode text based on input ids and attention mask

        Args:
            x (Tuple[Tensor, Tensor]): input ids and attention mask

        Returns:
            Tensor: encoded text
        """
        return self.encode(x)

    def construct_normalizer(self, normalize: bool) -> Callable:
        """Construct normalize function for output representations

        Args:
            normalize (bool): whether to normalize the output embedding

        Returns:
            Callable: normalize function 
        """
        if normalize:
            return torch.nn.functional.normalize
        else:
            return lambda x : x

    def construct_transformer(self, *args, **kwargs: dict) -> NoReturn:
        """
        Construct the transformer
        """
        raise NotImplementedError("")

    def construct_pooler(self, *args, **kwargs: dict) -> NoReturn:
        """
        Construct pooler to reduce output of Transformer layers
        """
        return lambda x: x

    @property
    def repr_dims(self) -> NoReturn:
        """
        The dimensionality of output/embedding space
        """
        raise NotImplementedError("")

    @property
    def _pooler(self) -> str:
        return self.__pooler

    @property
    def _vocabulary(self) -> NoReturn:
        raise NotImplementedError("")

    @property
    def _normalize(self) -> str:
        return self.__normalize

    @property
    def config(self) -> str:
        """
        The dimensionality of output/embedding space
        """
        return f"V: {self._vocabulary}; D: {self.repr_dims}; Normalize: {self._normalize}; Pooler: {self._pooler}"


