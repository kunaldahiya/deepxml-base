from . import xdataset, sdataset
from .dataset_base import DatasetTensor, DatasetBase


def _construct_dataset_class(sampling_t: str):
    """
    Return the dataset class

    Arguments:
    ----------
    sampling_t: str
        - implicit: (no-explicit negatives) in-batch sampling
        - explicit: explicitly sample negatives
        - brute: use all negatives
    """
    # assumes sampling is true
    if sampling_t == 'implicit':
        return sdataset.DatasetIS
    elif sampling_t == 'explicit':
        return sdataset.DatasetES
    elif sampling_t == 'brute':
        return xdataset.Dataset
    else:
        return DatasetTensor


def construct_dataset(data_dir,
                      fname=None,
                      data=None,
                      mode='train',
                      sampling_t='brute',
                      normalize_features=True,
                      normalize_labels=True,
                      keep_invalid=False,
                      feature_type='sparse',
                      max_len=-1,
                      **kwargs):    

    if fname is None:
        fname = {'f_features': None,
                 'f_labels': None,
                 'f_label_features': None}


    cls = _construct_dataset_class(sampling_t)
    return cls(data_dir=data_dir,
               **fname,
               data=data,
               mode=mode,
               max_len=max_len,
               normalize_features=normalize_features,
               normalize_labels=normalize_labels,
               keep_invalid=keep_invalid,
               feature_type=feature_type,
            )