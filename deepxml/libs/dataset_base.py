from typing import Optional, Any, Union, Tuple
from scipy.sparse import spmatrix
from numpy import ndarray
from xclib.data import FeaturesBase

import torch
from .features import construct as construct_f


class Dataset(torch.utils.data.Dataset):
    """Dataset for just the features
    * support for optional keyword arguments like max_len

    Args:
        data_dir (str): data directory
        fname (str): load data from this file
        data (Optional[Union[spmatrix, ndarray, Tuple]], optional): preloaded data.
            Defaults to None.
            * ndarray: dense features as numpy array
            * csr_matrix: sparse features as csr matrux
            * None: pre-loaded features not available; load from file
            * tuple: a tuple of ndarrays (useful in case of sequential features)
        normalize (bool, optional): normalize features. Defaults to False.
        _type (str, optional): type of features. Defaults to 'sparse'.
            * sparse
            * dense
            * sequential
    """
    def __init__(self,
                 data_dir: str,
                 fname: str,
                 data: Optional[Union[spmatrix, ndarray, Tuple]]=None,
                 normalize: bool=True,
                 _type: str='sparse',
                 **kwargs: Optional[Any]) -> None:
        self.data = self.construct(
            data_dir=data_dir,
            fname=fname,
            data=data,
            normalize=normalize,
            _type=_type,
            **kwargs)

    def construct(
            self,
            data_dir: str,
            fname: str,
            data: Optional[Union[spmatrix, ndarray, Tuple]]=None,
            normalize: bool=True,
            _type: str='sparse',
            **kwargs: Optional[Any]) -> FeaturesBase:
        return construct_f(data_dir, fname, data, normalize, _type, **kwargs)

    def __len__(self) -> int:
        return self.num_instances

    @property
    def num_instances(self) -> int:
        return self.data.num_instances

    def __getitem__(self, index: int) -> Tuple:
        """Get data for a given index
        Returns a tuple with features and index
        features can be:
        * ndarray in case of dense features
        * tuple of indices and weights in case of sparse features
        * tuple of indices and masks in case of sequential features
        """
        return (self.data[index], index)
