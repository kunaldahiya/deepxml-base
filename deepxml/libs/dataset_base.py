from typing import Optional, Any, Union, Tuple
from scipy.sparse import spmatrix
from numpy import ndarray
from xclib.data.features import FeaturesBase
from dataclasses import dataclass

import torch
import numpy as np
from .features import construct as construct_f
from .labels import construct as construct_l


@dataclass
class DataPoint:
    """
    DataPoint class represents a single data point in a dataset.

    Attributes:
        index (int): The index of the data point.
        x (torch.Tensor): The input features of the data point.
        y (Optional[Union[np.ndarray, Tuple]]): The target labels of the data point. Default is None.
        yf (Optional[Union[np.ndarray, Tuple]]): Additional target labels or features. Default is None.
    """
    index: int
    x: torch.Tensor
    y: Optional[Union[np.ndarray, Tuple]]=None
    y_neg: Optional[Union[np.ndarray, Tuple]]=None
    yf: Optional[Union[np.ndarray, Tuple]]=None
    yf_neg: Optional[Union[np.ndarray, Tuple]]=None


class DatasetBase(torch.utils.data.Dataset):
    """Dataset to load and use XML-Datasets

    Args:
        data_dir (str): data directory
        f_features (str): file containing features
            Support for sparse, dense and sequential features
        f_labels (str): file containing labels
            Support for sparse or dense
            * sparse will return just the positives
            * dense will return all the labels as a dense array
        f_label_features (Optional[str], optional): file containing label features.
          Defaults to None. Support for sparse, dense and sequential features
        data (dict, optional): preloaded features and labels.
          Defaults to {'X': None, 'Y': None, 'Yf': None}.
        mode (str, optional): train or test. Defaults to 'train'.
          may be useful in cases where different things are applied 
          to train or test set
        normalize_features (bool, optional): unit normalize? Defaults to True.
        normalize_lables (bool, optional): inf normalize? Defaults to False.
        feature_type (str, optional): feature type. Defaults to 'sparse'.
        label_type (str, optional): label type. Defaults to 'dense'.
        max_len (int, optional): max length. Defaults to -1.
    """
    def __init__(self,
                 data_dir: str,
                 f_features: str,
                 f_labels: str,
                 f_label_features: Optional[str]=None,
                 data: dict={'X': None, 'Y': None, 'Yf': None},
                 mode: str='train',
                 normalize_features: bool=True,
                 normalize_lables: bool=False,
                 feature_type: str='sparse',
                 label_type: str='dense',
                 max_len: int=-1,
                 *args: Optional[Any],
                 **kwargs: Optional[Any]) -> None:
        if data is None:
            data = {'X': None, 'Y': None, 'Yf': None}
        self.mode = mode
        self.features, self.labels, self.label_features = self.load_data(
            data_dir,
            f_features,
            f_labels,
            data,
            normalize_features,
            normalize_lables,
            feature_type,
            label_type,
            f_label_features,
            max_len)
        self.label_padding_index = self.num_labels

    def load_features(self, data_dir, fname, X,
                      normalize_features, feature_type, max_len):
        """Load features from given file
        Features can also be supplied directly
        """
        return construct_f(data_dir, fname, X,
                           normalize_features,
                           feature_type, max_len)

    def load_labels(self, data_dir, fname, Y, normalize_labels, label_type):
        """Load labels from given file
        Labels can also be supplied directly
        """
        return construct_l(data_dir, fname, Y, normalize_labels,
                             label_type)  # Pass dummy labels if required

    def load_data(self, data_dir, f_features, f_labels, data,
                  normalize_features=True, normalize_labels=False,
                  feature_type='sparse', label_type='dense',
                  f_label_features=None, max_len=32):
        """Load features and labels from file in libsvm format or pickle
        """
        features = self.load_features(
            data_dir, f_features, data['X'],
            normalize_features, feature_type, max_len)
        labels = self.load_labels(
            data_dir, f_labels, data['Y'], normalize_labels, label_type)
        label_features = None
        if f_label_features is not None or data["Yf"] is not None:
            label_features = self.load_features(
                data_dir, f_label_features, data['Yf'],
                normalize_features, feature_type, max_len)
        return features, labels, label_features

    @property
    def num_instances(self):
        return self.features.num_instances

    @property
    def num_features(self):
        return self.features.num_features

    @property
    def num_labels(self):
        return self.labels.num_labels

    def get_stats(self):
        """Get dataset statistics
        """
        return self.num_instances, self.num_features, self.num_labels

    def __len__(self):
        return self.num_instances

    @property
    def feature_type(self):
        return self.features._type

    def __getitem__(self, index: int) -> DataPoint:
        """Get features and labels for index
        Arguments
        ---------
        index: int
            data for this index
        """
        raise NotImplementedError("")


class DatasetSampling(DatasetBase):
    """Dataset to load and use XML-Datasets
    with shortlist
    
    Args:
        data_dir (str): data directory
        f_features (str): file containing features
            Support for sparse, dense and sequential features
        f_labels (str): file containing labels
            Support for sparse or dense
            * sparse will return just the positives
            * dense will return all the labels as a dense array
        f_label_features (Optional[str], optional): file containing label features.
          Defaults to None. Support for sparse, dense and sequential features
        data (dict, optional): preloaded features and labels.
          Defaults to {'X': None, 'Y': None, 'Yf': None}.
        mode (str, optional): train or test. Defaults to 'train'.
          may be useful in cases where different things are applied 
          to train or test set
        sampling_params (dict, optional): Parameters for sampler. Defaults to None.
        normalize_features (bool, optional): unit normalize? Defaults to True.
        normalize_lables (bool, optional): inf normalize? Defaults to False.
        feature_type (str, optional): feature type. Defaults to 'sparse'.
        label_type (str, optional): label type. Defaults to 'dense'.
        max_len (int, optional): max length. Defaults to -1.
    """

    def __init__(self,
                 data_dir: str,
                 f_features: str,
                 f_labels: str,
                 f_label_features: Optional[str]=None,
                 data: dict={'X': None, 'Y': None, 'Yf': None},
                 mode: str='train',
                 sampling_params: dict=None,
                 normalize_features: bool=True,
                 normalize_lables: bool=False,
                 feature_type: str='sparse',
                 label_type: str='dense',
                 max_len: int=-1,
                 *args: Optional[Any],
                 **kwargs: Optional[Any]) -> None:
        super().__init__(
            data_dir=data_dir,
            f_features=f_features,
            f_labels=f_labels,
            f_label_features=f_label_features,
            data=data,
            mode=mode, 
            normalize_features=normalize_features,
            normalize_lables=normalize_lables,
            feature_type=feature_type,
            max_len=max_len,
            label_type=label_type)
        self.sampler = self.construct_sampler(sampling_params)

    def construct_sampler(self, sampling_params):
        return None

    def indices_permutation(self) -> ndarray:
        return np.random.permutation(len(self.features))


class DatasetTensor(torch.utils.data.Dataset):
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

    def __getitem__(self, index: int) -> DataPoint:
        """Get data for a given index
        Returns a DataPoint instance with features and index
        features can be:
        * ndarray in case of dense features
        * tuple of indices and weights in case of sparse features
        * tuple of indices and masks in case of sequential features
        """
        return DataPoint(x=self.data[index], index=index)
