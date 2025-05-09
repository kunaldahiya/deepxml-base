from torch import Tensor, dtype
from numpy import ndarray
from typing import Iterator, Callable

import torch
import numpy as np
from functools import partial
from .datapoint import DataPoint
from scipy.sparse import csr_matrix
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
    if max_len == -1:
        return ind, mask
    _max = min(torch.max(torch.sum(mask, dim=1)), max_len)
    return ind[:, :_max], mask[:, :_max]


def pad_and_collate(
        x: Iterator[ndarray], 
        pad_val: int=0, 
        dtype: dtype=torch.float) -> Tensor:
    """A generalized function for padding batch using utils.rnn.pad_sequence
    * pad as per the maximum length in the batch
    * returns a collated tensor

    Args:
        x (Iterator[ndarray]): iterator over np.ndarray that needs to be converted to
            tensors and padded
        pad_val (int, optional): pad tensor with this value. Defaults to 0.
            will cast the value as per the data type
        dtype (dtype, optional): tensor should be of this type. 
            Defaults to torch.float.

    Returns:
        Tensor: a padded and collated tensor
    """
    return pad_sequence([torch.from_numpy(z) for z in x],
                        batch_first=True, padding_value=pad_val).type(dtype)


def collate_dense(
        x: Iterator[ndarray], 
        dtype: dtype=torch.float) -> Tensor:
    """Collate dense arrays

    Args:
        x (Iterator[ndarray]): iterator over np.ndarray that needs to be converted to
            tensors
        dtype (dtype, optional): tensor should be of this type. 
            Defaults to torch.float.

    Returns:
        Tensor: Collated dense tensor
    """
    x = list(x) # x can be an iterator
    if x[0].ndim == 1:
        return torch.from_numpy(np.stack(x)).type(dtype)
    else:
        return torch.from_numpy(np.concatenate(x)).type(dtype)


def collate_as_1d(x: Iterator, dtype: dtype) -> Tensor:
    """Collate and return a 1D tensor

    Args:
        x (Iterator): iterator over numbers 
        dtype (dtype): data type

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


def collate_sparse(
        batch: Iterator,
        pad_ind=0,
        pad_val=0,
        dtype_ind=torch.int,
        dtype_val=torch.float) -> dict:
    
    batch = list(batch)
    indices = pad_and_collate(
        map(lambda z: z[0], batch), 
        pad_ind,
        dtype_ind)
    weights = pad_and_collate(
        map(lambda z: z[1], batch),
        pad_val,
        dtype_val)
    return weights, indices


def collate_sequential(batch: Iterator[tuple], max_len) -> tuple:
    """Collate sequential features with indices and mask
    * Expects a Iterator over tuples of ind and mask
    * Must be of same length 
    * TODO: Implement padding

    Args:
        batch (Iterator[tuple]): Iterator over tuples of ind and mask

    Returns:
        tuple: _description_
    """
    batch = list(batch)
    indices = collate_dense(map(lambda z: z[0], batch), dtype=torch.long)
    mask = collate_dense(map(lambda z: z[1], batch), dtype=torch.long)
    return clip_batch_lengths(indices, mask, max_len)


def collate_brute(batch: Iterator[ndarray]) -> tuple:
    """Collate over iterator of dense/brute labels (all labels are considered)

    Args:
        batch (Iterator[ndarray]): an iterator over relevance vectors
            * Each vector is of length as number of labels

    Returns:
        tuple: collated data
    """
    return collate_dense(batch), None, None


def collate_implicit(batch: Iterator) -> tuple:
    """Collate labels over iterator for implicit samling (no negatives)
    * It does not select unique labels so that we don't observe any shape
      mis-matchs when labels embeddings are involved (they are stacked together)
    
    Args:
        batch (Iterator): an iterator over positives. It can be over: 
            * ndarrays: will consider all the labels 
              useful where positives are not sampled 
            * tupe of ndarray: (sampled positives, all_positives) 
              - useful when positives are sampled to save on memory and compute
              - the sampled labels will be considered for label pool and all 
                labels will only be used to avoid False Negatives
    Returns:
        tuple: collated data
    """
    lens = []
    lens_sampled = []
    batch_labels = []
    sampled_pos_indices = []
    l_max = -1
    for item in batch:
        #TODO: See if there is a more efficient way to do it
        if isinstance(item, tuple):
            _s, _a = item
        else:
            _s = _a = item
        l_max = max(l_max, max(_a, default=-1))
        lens.append(len(_a))
        batch_labels.append(_a)
        lens_sampled.append(len(_s))
        sampled_pos_indices.append(_s)

    batch_size = len(batch_labels)
    l_max += 1 # indices start from zero

    rows = np.repeat(range(batch_size), lens)
    cols = np.concatenate(batch_labels, axis=None)
    data = np.ones((len(rows), ), dtype='bool')
    A = csr_matrix((data, (rows, cols)), shape=(batch_size, l_max))

    cols = np.concatenate(sampled_pos_indices, axis=None)
    rows = np.arange(len(cols))
    data = np.ones((len(rows), ), dtype='bool')
    B = csr_matrix((data, (cols, rows)), shape=(l_max, len(cols)))
    batch_selection = (A @ B).toarray().astype('float32')
    return torch.from_numpy(batch_selection), torch.LongTensor(cols), None


class collate():
    """
    A generic class to handle different features, classifiers and sampling
    """
    def __init__(
            self, 
            in_feature_t: str = 'sequential', 
            sampling_t: str = 'implicit', 
            op_feature_t: str = None,
            max_len: int = -1) -> None:
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
            max_len (str, int): Clips sequential features. Defaults to -1.
                * No action if it is -1
            """
        self.max_len = max_len
        self.collate_ip_features = self.construct_feature_collator(in_feature_t)
        self.collate_op_features = self.construct_feature_collator(op_feature_t)
        self.collate_labels = self.construct_label_collator(sampling_t)

    def construct_feature_collator(self, _type: str) -> Callable:
        if _type == "dense":
            return collate_dense
        elif _type == "sparse":
            return collate_sparse
        elif _type == "sequential":
            return partial(collate_sequential, max_len=self.max_len)
        else:
            return None

    def construct_label_collator(self, sampling_t) -> Callable:
        if sampling_t == 'implicit':
            return collate_implicit
        elif sampling_t == 'explicit':
            raise NotImplementedError("")
        elif sampling_t == 'brute':
            return collate_brute
        else:
            return None

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