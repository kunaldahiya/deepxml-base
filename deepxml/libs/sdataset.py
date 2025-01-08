from typing import Optional, Any
from argparse import Namespace
from numpy import ndarray

import numpy as np
from .dataset_base import DatasetBase, DataPoint, DatasetSampling
from .shortlist import ClusteringIndex
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

    def construct_sampler(self, params: Optional[Namespace]=None) -> None:
        if params is not None:
            depth = compute_depth_of_tree(
                self.__len__(),
                params.init_cluster_size)
            return ClusteringIndex(
                num_instances=self.__len__(),
                num_clusters=2**depth,
                num_threads=params.threads,
                curr_steps=params.curr_epochs)

    def indices_permutation(self) -> ndarray:
        if self.sampler is None:
            return super().indices_permutation()
        clusters = np.arange(self.sampler.num_clusters)
        np.random.shuffle(clusters)
        indices = []
        for it in clusters:
            indices.extend(self.sampler.query[it])
        return np.array(indices)

    def update_state(self, *args):
        self.sampler.update_state()

    def update_sampler(self, *args):
        """Update negative sampler
        """
        self.sampler.update(*args)

    def __getitem__(self, index: int) -> DataPoint:
        """Get the data at index"""
        pos_ind, _ = self.labels[index]
        if self.n_pos == -1:
            Yf = None if self.label_features is None \
                else self.label_features[pos_ind]
        else:
            sampled_pos_ind = np.random.choice(pos_ind, size=self.n_pos)
            Yf = None if self.label_features is None \
                else self.label_features[sampled_pos_ind]
            pos_ind = (sampled_pos_ind, pos_ind)
        return DataPoint(
            x=self.features[index],
            y=pos_ind,
            yf=Yf, 
            index=index)
