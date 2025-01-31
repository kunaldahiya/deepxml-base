from typing import Optional, Any
from argparse import Namespace
from numpy import ndarray

import numpy as np
from .dataset_base import DatasetBase, DataPoint, DatasetSampling
from .sampler import ClusteringIndex, ANNIndex
from .utils import compute_depth_of_tree


class Dataset(DatasetBase):
    """Dataset to load and use XML-Datasets with sparse 
       classifiers or embeddings
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
                 label_type: str='sparse',
                 max_len: int=-1,
                 *args: Optional[Any],
                 **kwargs: Optional[Any]) -> None:
        super().__init__(data_dir=data_dir,
                         f_features=f_features,
                         data=data,
                         f_label_features=f_label_features,
                         f_labels=f_labels,
                         max_len=max_len,
                         normalize_features=normalize_features,
                         normalize_lables=normalize_lables,
                         feature_type=feature_type,
                         label_type=label_type,
                         mode=mode
                        )        

    def __getitem__(self, index: int) -> DataPoint:
        """Get the data at index"""
        pos_labels, _ = self.labels[index]
        return DataPoint(
            x=self.features[index],
            y=pos_labels,
            yf=None if self.label_features is None \
                else self.label_features[pos_labels], 
            index=index)


class DatasetIS(DatasetSampling):
    """Dataset to load and use XML-Datasets with sparse 
       classifiers or embeddings
       * Use with in-batch sampling
    """
    def __init__(self,
                 data_dir: str,
                 f_features: str,
                 f_labels: str,
                 sampling_params: Optional[Namespace]=None,
                 f_label_features: Optional[str]=None,
                 data: dict={'X': None, 'Y': None, 'Yf': None},
                 mode: str='train',
                 normalize_features: bool=True,
                 normalize_lables: bool=False,
                 feature_type: str='sparse',
                 label_type: str='sparse',
                 max_len: int=-1,
                 n_pos: int=1,
                 *args: Optional[Any],
                 **kwargs: Optional[Any]) -> None:
        """
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
            sampling_params (Namespace, optional): Parameters for sampler. Defaults to None.
            n_pos (int, optional): Number of positives for each item
                * n_pos specified in sampling_params will take priority
            normalize_features (bool, optional): unit normalize? Defaults to True.
            normalize_lables (bool, optional): inf normalize? Defaults to False.
            feature_type (str, optional): feature type. Defaults to 'sparse'.
            label_type (str, optional): label type. Defaults to 'dense'.
            max_len (int, optional): max length. Defaults to -1.
        """
        super().__init__(data_dir=data_dir,
                         f_features=f_features,
                         data=data,
                         sampling_params=sampling_params,
                         f_label_features=f_label_features,
                         f_labels=f_labels,
                         max_len=max_len,
                         normalize_features=normalize_features,
                         normalize_lables=normalize_lables,
                         feature_type=feature_type,
                         label_type=label_type,
                         mode=mode
                        )
        self.n_pos = sampling_params.n_pos \
            if hasattr(sampling_params, 'n_pos') else n_pos

    def step(self, *args, **kwargs):
        # the negative sampler should take care of updates and all
        self.sampler.step(*args, **kwargs)

    def construct_sampler(self, params: Optional[Namespace]=None) -> None:
        if params is not None:
            depth = compute_depth_of_tree(
                self.__len__(),
                params.init_cluster_size)
            return ClusteringIndex(
                num_items=self.__len__(),
                num_clusters=2**depth,
                num_threads=params.threads,
                update_steps=params.update_steps,
                curr_steps=params.curr_steps)

    def indices_permutation(self) -> ndarray:
        if self.sampler is None:
            return super().indices_permutation()
        return self.sampler.indices_permutation()

    def update_sampler(self, *args):
        """Update negative sampler
        """
        self.sampler.update(*args)

    def _sample_y_and_yf(self, ind, n=-1):
        if n == -1 or len(ind) < n or len(ind) == 0:
            sampled_ind = ind
        else:
            sampled_ind = np.random.choice(ind, size=n), 
        Yf = None if self.label_features is None \
            else self.label_features[sampled_ind]
        return sampled_ind, Yf

    def __getitem__(self, index: int) -> DataPoint:
        """Get the data at index"""
        pos_ind, _ = self.labels[index]
        sampled_ind, Yf = self._sample_y_and_yf(pos_ind, self.n_pos)
        return DataPoint(
            x=self.features[index],
            y=(sampled_ind, pos_ind),
            yf=Yf, 
            index=index)


class DatasetES(DatasetSampling):
    """Dataset to load and use XML-Datasets with sparse 
       classifiers or embeddings
       * Use with explicit negative sampling
       * Builds an ANNS Index over the label representations
    """
    def __init__(self,
                 data_dir: str,
                 f_features: str,
                 f_labels: str,
                 sampling_params: Optional[Namespace]=None,
                 f_label_features: Optional[str]=None,
                 data: dict={'X': None, 'Y': None, 'Yf': None},
                 mode: str='train',
                 normalize_features: bool=True,
                 normalize_lables: bool=False,
                 feature_type: str='sparse',
                 label_type: str='sparse',
                 max_len: int=-1,
                 n_pos: int=1,
                 n_neg: int=1,
                 *args: Optional[Any],
                 **kwargs: Optional[Any]) -> None:
        """
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
            n_pos (int, optional): Number of positives for each item
            n_neg (int, optional): Number of negatives for each item
            normalize_features (bool, optional): unit normalize? Defaults to True.
            normalize_lables (bool, optional): inf normalize? Defaults to False.
            feature_type (str, optional): feature type. Defaults to 'sparse'.
            label_type (str, optional): label type. Defaults to 'dense'.
            max_len (int, optional): max length. Defaults to -1.
        """
        super().__init__(data_dir=data_dir,
                         f_features=f_features,
                         data=data,
                         sampling_params=sampling_params,
                         f_label_features=f_label_features,
                         f_labels=f_labels,
                         max_len=max_len,
                         normalize_features=normalize_features,
                         normalize_lables=normalize_lables,
                         feature_type=feature_type,
                         label_type=label_type,
                         mode=mode
                        )
        self.n_pos = n_pos
        self.n_neg = n_neg        

    def _default_params(self):
        return Namespace(
            method='hnswlib',
            num_neighbours=50,
            M=50,
            efC=50,
            num_threads=6,
            space='cosine')

    def construct_sampler(self, params: Optional[Namespace]=None) -> None:
        params = self._default_params() if params is None else params
        return ANNIndex(
            num_items=self.num_labels,
            method=params.method, 
            num_neighbours=params.num_neighbours, 
            M=params.M, 
            efC=params.efC, 
            num_threads=params.num_threads, 
            space=params.space
        )

    def update_sampler(self, *args):
        """Update negative sampler
        """
        self.sampler.update(*args)

    def _sample_and_yf(self, ind, n=-1):
        if n == -1:
            sampled_ind = ind
        else:
            sampled_ind = np.random.choice(ind, size=n)
        Yf = None if self.label_features is None \
            else self.label_features[sampled_ind]
        return sampled_ind, Yf


    def __getitem__(self, index: int) -> DataPoint:
        """Get the data at index"""
        # TODO: Fix the following
        # - Ensure that the negatives do not contain explicit positives 
        pos_ind, Yf = self._sample_and_yf(self.labels[index][0], self.n_pos)
        neg_ind, Yf_neg = self._sample_and_yf(self.sampler[index][0], self.n_neg)
        return DataPoint(
            x=self.features[index],
            y=pos_ind,
            yf=Yf, 
            y_neg=neg_ind,
            yf_neg=Yf_neg,
            index=index)
