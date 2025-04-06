from torch.utils.data import Dataset

from argparse import Namespace
from .dataset_factory import DatasetFactory
from .dataset_base import DatasetTensor


def _construct_dataset_class(sampling_t: str, dataset_factory: dict) -> Dataset:
    """
    Return the dataset class

    Arguments:
    ----------
    sampling_t: str
        - implicit: (no-explicit negatives) in-batch sampling
        - explicit: explicitly sample negatives
        - brute: use all negatives
        - tensor: no negative sampling (e.g., iterate over data points)
    """
    return dataset_factory.get(sampling_t, DatasetTensor)


def construct_dataset(data_dir: str,
                      fname: str | dict=None,
                      data: dict=None,
                      mode: str='train',
                      sampling_t: str='brute',
                      normalize_features: bool=True,
                      normalize_labels: bool=True,
                      sampling_params: Namespace=Namespace(),
                      keep_invalid: bool=False,
                      feature_t: str='sparse',
                      max_len: int=-1,
                      dataset_factory: dict=DatasetFactory,
                      **kwargs):    
    if fname is None:
        fname = {'f_features': None,
                 'f_labels': None,
                 'f_label_features': None,
                 'f_label_filter': None}

    if hasattr(sampling_params, 'type'):
        assert sampling_t == sampling_params.type, \
            "type in sampling_params must match sampling_t"

    cls = _construct_dataset_class(sampling_t, dataset_factory)
    return cls(data_dir=data_dir,
               **fname,
               data=data,
               mode=mode,
               max_len=max_len,
               sampling_params=sampling_params,
               normalize_features=normalize_features,
               normalize_labels=normalize_labels,
               keep_invalid=keep_invalid,
               feature_t=feature_t,
               **kwargs
            )