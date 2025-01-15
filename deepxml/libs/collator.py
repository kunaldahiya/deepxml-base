from torch import Tensor, tensortype
from numpy import ndarray
from typing import Iterator

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


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