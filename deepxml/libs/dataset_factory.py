from . import xdataset, sdataset
from .dataset_base import DatasetTensor


DatasetFactory = {
    'implicit': sdataset.DatasetIS,
    'explicit': sdataset.DatasetES,
    'brute': xdataset.Dataset,
    'default': DatasetTensor,
    'tensor': DatasetTensor
}