from torch import Tensor, tensortype
from numpy import ndarray
from typing import Iterator, Callable


import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .datapoint import DataPoint


def clip_batch_lengths(ind: Tensor, mask: Tensor, max_len: int=1024) -> tuple:
    """Clip indices and mask as per given max length

    Args:
        ind (Tensor): indices
        mask (Tensor): mask (useful for padding)
        max_len (int, optional): Cut after this length. Defaults to 1024.

    Returns:
        tuple: Clipped ind and mask
    """
    _max = min(torch.max(torch.sum(mask, dim=1)), max_len)
    return ind[:, :_max], mask[:, :_max]


def pad_and_collate(
        x: Iterator[ndarray], 
        pad_val: int=0, 
        dtype: tensortype=torch.FloatTensor) -> Tensor:
    """A generalized function for padding batch using utils.rnn.pad_sequence
    * pad as per the maximum length in the batch
    * returns a collated tensor

    Args:
        x (Iterator[ndarray]): iterator over np.ndarray that needs to be converted to
            tensors and padded
        pad_val (int, optional): pad tensor with this value. Defaults to 0.
            will cast the value as per the data type
        dtype (tensortype, optional): tensor should be of this type. 
            Defaults to torch.FloatTensor.

    Returns:
        Tensor: a padded and collated tensor
    """
    return pad_sequence([torch.from_numpy(z) for z in x],
                        batch_first=True, padding_value=pad_val).type(dtype)


def collate_dense(
        x: Iterator[ndarray], 
        dtype: tensortype=torch.FloatTensor) -> Tensor:
    """Collate dense arrays

    Args:
        x (Iterator[ndarray]): iterator over np.ndarray that needs to be converted to
            tensors
        dtype (tensortype, optional): Output type. Defaults to torch.FloatTensor.

    Returns:
        Tensor: Collated dense tensor
    """
    return torch.stack([torch.from_numpy(z) for z in x], 0).type(dtype)


def collate_as_1d(x: Iterator, dtype: tensortype) -> Tensor:
    """Collate and return a 1D tensor

    Args:
        x (Iterator): iterator over numbers 
        dtype (tensortype): datatype

    Returns:
        Tensor: Collated 1D tensor
    """
    return torch.from_numpy(np.concatenate(list(x))).type(dtype)


def collate_as_np_1d(x: Iterator, dtype: str) -> ndarray:
    """Collate and return a 1D ndarray

    Args:
        x (Iterator): iterator over numbers
        dtype (str): datatype

    Returns:
        ndarray: Collated 1D tensor
    """
    return np.fromiter(x, dtype=dtype)


def get_iterator(x: Iterator[DataPoint], key: str=None) -> Iterator:
    """
    Returns an iterator over the given data points. 
    * If a key is provided: the iterator will yield the value of the key
      attribute for each data point. 
    * Otherwise, it will yield the data points themselves.

    Args:
        x (Iterator[DataPoint]): An iterator over data points.
        key (str, optional): The key attribute to extract 
            from each data point. Defaults to None.

    Returns:
        Iterator: An iterator over the data points or their key attributes.
    """
    if key is None:
        return map(lambda z: z, x)
    else:
        return map(lambda z: getattr(z, key), x)


def collate_sparse(batch: Iterator) -> dict:
    raise NotImplementedError("")


def collate_sequential(batch: Iterator) -> dict:
    raise NotImplementedError("")


def collate_brute(batch: Iterator):
    return collate_dense(batch), None, None


def collate_implicit(batch: Iterator):
    raise NotADirectoryError("")


class collate():
    """
    A generic class to handle different features, classifiers and sampling
    """
    def __init__(
            self, 
            in_feature_t: str = 'sequential', 
            sampling_t: str = 'implicit', 
            op_feature_t: str = None) -> None:
        """
        Args:
            in_feature_t (str, optional): feature type of input items. 
                Defaults to 'sequential'.   
                * dense: dense features. Such as pre-computed features.
                * sparse: sparse features. Indices and weights  
                * sequential: Sequential features
            sampling_t (str, optional): sampling type. Defaults to 'implicit'.
                * implicit: in-batch sampling (use positives of other documents) 
                * explicit: explicit negatives for each labels 
                * brute: 1-vs-all classifiers
            op_feature_t (str, optional): Output feature type. Defaults to None.
                Defaults to 'sequential'.   
                * dense: dense features. Such as pre-computed features.
                * sparse: sparse features. Indices and weights  
                * sequential: Sequential features
            """
        self.collate_ip_features = self.construct_feature_collator(in_feature_t)
        self.collate_op_features = self.construct_feature_collator(op_feature_t)
        self.collate_labels = self.construct_label_collator(sampling_t)

    def construct_feature_collator(self, _type: str) -> Callable:
        if _type == "dense":
            return collate_dense
        elif _type == "sparse":
            return collate_sparse
        elif _type == "sequential":
            return collate_sequential
        else:
            return None

    def construct_label_collator(self, classifier_t, sampling_t) -> Callable:
        if classifier_t is None:
            return None

        if sampling_t == 'implicit':
            return collate_implicit
        elif sampling_t == 'explicit':
            raise NotImplementedError("")
        elif sampling_t == 'brute':
            return collate_brute
        else:
            raise NotImplementedError("")

    def __call__(self, batch):
        data = {}
        data['batch_size'] = torch.tensor(len(batch), dtype=torch.int32)
        data['X'] = self.collate_ip_features(get_iterator(batch, 'x'))
        data['indices'] = torch.LongTensor([it.index for it in batch])
        if self.collate_labels is not None: # labels are availabels
            data['Y'], data['Y_s'], data['Y_mask'] = self.collate_labels(
                get_iterator(batch, 'y'))
        if self.collate_op_features is not None: # label features are available
            data['Z'] = self.collate_op_features(get_iterator(batch, 'yf'))
        return data