from typing import Optional, Callable, Union, Tuple, NoReturn, Any, List

import re
import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import AutoModel, AutoConfig, PreTrainedModel
from functools import partial
from operator import itemgetter


def mean_pooling(emb: Tensor, mask: Tensor) -> Tensor:
    """Mean pooling to get the sentence embeddings

    Args:
        emb (Tensor): Final embeddings from a transformer layer
            shape (batch_size, seq_len, embedding_dim)
        mask (Tensor): To mask out padded values
            (batch_size, seq_len)

    Returns:
        Tensor: mean pooled sentence representation 
            (batch_size, embedding_dim)
    """
    mask = mask.unsqueeze(-1).expand(emb.size()).float()
    sum_emb = torch.sum(emb * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    return sum_emb / sum_mask


class TransformerEncoderBag(torch.nn.Module):
    def __init__(
            self, 
            d_model: int=512, 
            n_head: int=1,
            dim_feedforward: int=2048,
            dropout: float=0.1,
            activation: str='gelu',
            norm_first: bool=False):
        """A wrapper over TransformerEncoderLayer to apply self-attention
        over a bag-of-embeddings (order is ignored)

        Args:
            d_model (int, optional): Dim of input embeddings. Defaults to 512.
            n_head (int, optional): number of heads. Defaults to 1.
            dim_feedforward (int, optional): Defaults to 2048.
                Dim of internal linear layer.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            activation (str, optional): Activation function. Defaults to 'gelu'.
            norm_first (bool, optional): first normalize? Defaults to False.
        """
        super(TransformerEncoderBag, self).__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model, n_head,
            dim_feedforward,
            dropout,
            activation=activation,
            norm_first=norm_first)

    def forward(self, x: Tensor | List[Tensor]) -> Tensor:
        """Apply self attention and return a single vector for each instance

        Args:
            x (Tensor | List[Tensor]): 
                - List is over tensors of size (N, E). 
                    These will be stacked along a new axis and then pooled
                - Tensor: Already stacked tensor with size (S, N, E)
                    Note that here batch is not the first dimension

        Returns:
            Tensor: Pooled Tensor with size (N, E)
        """
        if isinstance(x, list):
            x = torch.stack(x) 
        # src is expected to have shape (S, N, E) where S is the sequence length,
        # N is the batch size, and E is the embedding dimension
        output = self.encoder_layer(x) # output has shape (S, N, E)
        return torch.mean(output, 0).squeeze()


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
            **kwargs: Optional[Any]):
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
            **kwargs: Optional[Any]) -> NoReturn:
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

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tensor:
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

    def construct_transformer(
            self,
            *args: Optional[Any],
            **kwargs: Optional[Any]) -> NoReturn:
        """
        Construct the transformer
        """
        raise NotImplementedError("")

    def construct_pooler(
            self,
            *args: Optional[Any],
            **kwargs: Optional[Any]) -> Callable:
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


class STransformer(BaseTransformer):
    """Create Transformers using Sentence Bert Library
        * Use default pooler of trained model (yields better results)
        * Use HTransformer if you want to customize pooler
        * mean pooler is equivalent to using mean_pooling on 
        HTransformer's last_hidden_state followed by an optional normalize layer


    Args:
        transformer (Optional[Union[str, SentenceTransformer]], optional):
            transformer string or object 
            * Defaults to 'bert-base-uncased'.
        normalize (Optional[bool], optional): normalize output? 
            Defaults to False.
    """
    def __init__(
            self,
            transformer: Optional[Union[str, SentenceTransformer]]='bert-base-uncased',
            normalize: Optional[bool]=False,
            **kwargs: Optional[Any]):
        super(STransformer, self).__init__(transformer, None, normalize)

    def construct_transformer(
            self,
            transformer: Union[str, SentenceTransformer],
            **kwargs: Optional[Any]) -> SentenceTransformer:
        if isinstance(transformer, str):
            return SentenceTransformer(transformer)
        else:
            return transformer

    def encode(self, x: Tuple[Tensor, Tensor]) -> Tensor:
        """Forward pass to encode text based on input ids and attention mask

        Args:
            x (Tuple[Tensor, Tensor]): input ids and attention mask

        Returns:
            Tensor: encoded text
        """
        ids, mask = x
        out = self.transform({'input_ids': ids, 'attention_mask': mask})
        return self.normalize(out['sentence_embedding'])

    @property
    def repr_dims(self) -> int:
        return self.transform[1].word_embedding_dimension

    @property
    def _pooler(self) -> dict:
        keys = [x for x in self.transform[1].__dict__['config_keys']\
             if re.match("pooling*", x)]        
        return dict(zip(keys, itemgetter(*keys)(self.transform[1].__dict__)))

    @property
    def _vocabulary(self) -> int:
        return self.transform[0].vocab_size


class HTransformer(BaseTransformer):
    """Create Transformers using Huggingface library. Support for:
     * custom pooling
     * concatenation of layers
    
    Args:
        transformer (Optional[Union[str, PreTrainedModel]], optional): 
            transformer from Huggingface. Defaults to 'bert-base-uncased'.
        normalize (Optional[bool], optional): return normalized outputs 
            or as it is. Defaults to False.
        pooler (Optional[str], optional): pooling function. Defaults to None.
            method to reduce the output of transformer layers
            * Support for mean, None (identity), concat and cls
        c_layers (List[int], optional): concatenate rep. Defaults to [-1, -4].
            concatenate these layers when pooler is concat (ignored otherwise)
    """
    def __init__(
            self,
            transformer: Optional[Union[str, PreTrainedModel]]='bert-base-uncased',
            normalize: Optional[bool]=False,
            pooler: Optional[str]=None,
            c_layers: List[int]=[-1, -4]):
        if pooler != "concat":
            c_layers = None
        super(HTransformer, self).__init__(
            transformer, pooler, normalize, c_layers=c_layers)
        self._c_layers = c_layers

    def construct_transformer(
            self,
            transformer: Union[str, PreTrainedModel],
            c_layers: List[int]) -> PreTrainedModel:
        """Construct transformer

        Args:
            transformer (Union[str, PreTrainedModel]): transformer (name or object)
            c_layers (List[int]): layers to concatenate

        Returns:
            PreTrainedModel: Transformer object
        """
        output_hidden_states = True if isinstance(c_layers, list) else True
        if isinstance(transformer, str):
            config = AutoConfig.from_pretrained(
                transformer, 
                output_hidden_states=output_hidden_states)
            return AutoModel.from_pretrained(transformer, config=config)
        else:
            return transformer

    def encode(self, x: Tuple[Tensor, Tensor]) -> Tensor:
        """Encode text based on input ids and attention mask

        Args:
            x (Tuple[Tensor, Tensor]): input ids and attention mask

        Returns:
            Tensor: encoded text
        """
        ids, mask = x
        out = self.transform(input_ids=ids, attention_mask= mask)
        return self.normalize(self.pooler(out, mask))

    def construct_pooler(self, pooler: str, c_layers: List[int]) -> Callable:
        """Construct function to reduce or pool

        Args:
            pooler (str): pooling name
            c_layers (List[int]): layers to concatenate 
                (valid only when pooler is concat)

        Returns:
            Callable: pooling function
        """
        if pooler is None:
                return lambda x, _: x['last_hidden_state']
        elif pooler == 'concat':
            assert isinstance(c_layers, list), "list is expected for concat"
            def f(x: Tensor, m: Tensor, c_l: List[int]) -> Tensor:
                r = []
                for l in c_l:
                    r.append(
                        mean_pooling(x['hidden_states'][l], m))
                return torch.hstack(r)
            return partial(f, c_l=c_layers)
        elif pooler == 'mean':
            def f(x: Tensor, m: Tensor) -> Tensor:
                return mean_pooling(x['last_hidden_state'], m)
            return f
        elif pooler == 'cls':
            def f(x: Tensor, *args) -> Tensor:
                return x['last_hidden_state'][:, 0]
            return f
        else:
            print(f'Unknown pooler type encountered: {pooler}')

    @property
    def repr_dims(self) -> int:
        """
        The dimensionality of output/embedding space
        """
        d = self.transform.embeddings.word_embeddings.embedding_dim
        if self._pooler == "concat":
            return d * len(self._c_layers) 
        else:
            return d

    @property
    def config(self):
        return self.transform.config
