from typing import Optional, Any, Union, Tuple
from scipy.sparse import spmatrix
from numpy import ndarray
from xclib.data.features import FeaturesBase

import os
import numpy as np
from xclib.data.features import SparseFeatures
from xclib.data.features import DenseFeatures as _DenseFeatures


def construct(
        data_dir: str,
        fname: str,
        X: Optional[Union[spmatrix, ndarray, Tuple]]=None,
        normalize: bool=False,
        _type: str='sparse',
        max_len:int =-1,
        **kwargs: Optional[Any]) -> FeaturesBase:
    """Construct feature class based on given parameters

    Args:
        data_dir (str): data directory
        fname (str): load data from this file
        X (Optional[Union[spmatrix, ndarray, Tuple]], optional): preloaded data.
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
        max_len (int, optional): max length in seq features. Defaults to -1.

    Raises:
        NotImplementedError: if feature type is unknown

    Returns:
        FeaturesBase: feature object
    """
    if _type == 'sparse':
        return _SparseFeatures(data_dir, fname, X, normalize)
    elif _type == 'dense':
        return DenseFeatures(data_dir, fname, X, normalize)
    elif _type == 'sequential':
        return SeqFeatures(data_dir, fname, X, max_len)
    else:
        raise NotImplementedError("Unknown feature type")


class SeqFeatures(SparseFeatures):
    """Class for sequential features

    Args:
        data_dir (str): data directory
        fname (str): load data from this file
        X (Optional[Tuple[ndarray, ndarray]], optional): preloaded features.
            Defaults to None.
            * None: pre-loaded features not available; load from file
            * tuple: a tuple of ndarrays (useful in case of sequential features)
        max_len (int, optional): _description_. Defaults to -1.
    """
    def __init__(
            self,         
            data_dir: str,
            fname: str,
            X: Optional[Tuple[ndarray, ndarray]]=None,
            max_len:int =-1) -> None:
        super().__init__(data_dir, fname, X)
        self.max_len = max_len

    def load(
            self,
            data_dir: str,
            fname: str,
            X: Optional[Tuple[ndarray, ndarray]]=None
            ) -> Tuple[ndarray, ndarray]:
        """Load sequential features

        Args:
            data_dir (str): data directory
            fname (str): load data from this file
            X (Optional[Tuple[ndarray, ndarray]], optional): preloaded data.
                Defaults to None.
                * None: pre-loaded features not available; load from file
                * tuple: a tuple of ndarrays (useful in case of sequential features)

        Returns:
            Tuple[ndarray, ndarray]: tokenized array and mask
        """
        if X is not None:
            return X
        else:
            f_ids, f_mask = fname.split(",")
            X = np.load(
                os.path.join(data_dir, f_ids),
                mmap_mode='r')
            X_mask = np.load(
                os.path.join(data_dir, f_mask),
                mmap_mode='r')
            return X, X_mask

    @property
    def data(self) -> Tuple[ndarray, ndarray]:
        return self.X

    def __getitem__(self, index: int) -> Tuple[ndarray, ndarray]:
        if self.max_len > 0:
            return self.X[0][index][:self.max_len], \
                self.X[1][index][:self.max_len]
        else:
            return (self.X[0][index], self.X[1][index])

    @property
    def num_instances(self) -> int:
        return len(self.X[0])

    @property
    def num_features(self) -> int:
        return -1

    @property
    def _type(self) -> str:
        return 'sequential'

    @property
    def _params(self) -> dict:
        return {'max_len': self.max_len,
                'feature_type': self._type,
                '_type': self._type}


class DenseFeatures(_DenseFeatures):
    @property
    def _type(self):
        return 'dense'

    @property
    def _params(self):
        return {'feature_type': self._type,
                '_type': self._type}


class _SparseFeatures(SparseFeatures):
    """Class for sparse features
    * Difference: treat 0 as padding index

    Args:
        data_dir (str): data directory
        fname (str): load data from this file
        X (Optional[spmatrix], optional): pre-loaded features. Defaults to None.
        normalize (bool, optional): unit normalize features. Defaults to False.
    """
    def __init__(
            self,
            data_dir: str,
            fname: str,
            X: Optional[spmatrix]=None,
            normalize: bool=False) -> None:
        super().__init__(data_dir, fname, X, normalize)

    def __getitem__(self, index: int) -> Tuple[ndarray, ndarray]:
        # Treat idx:0 as Padding
        x = self.X[index].indices + 1
        w = self.X[index].data
        return x, w

    @property
    def _type(self) -> str:
        return 'sparse'

    @property
    def _params(self) -> dict:
        return {'feature_type': self._type,
                '_type': self._type}