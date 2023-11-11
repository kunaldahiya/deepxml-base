from typing import Optional, Any

from .dataset_base import DatasetBase


class Dataset(DatasetBase):
    """Dataset to load and use XML-Datasets with OVA classifiers

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

    def __getitem__(self, index):
        """Get a label at index"""
        doc_ft = self.features[index]
        ind, wts = self.labels[index]
        return (doc_ft, (ind, wts), index)
 