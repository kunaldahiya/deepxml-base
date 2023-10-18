from typing import Optional
from scipy.sparse import spmatrix

from xclib.data.labels import DenseLabels, SparseLabels, LabelsBase


def construct(
        data_dir: str,
        fname: str,
        Y: Optional[spmatrix]=None,
        normalize: bool=False,
        _type: str='sparse') -> LabelsBase:
    """Construct label class based on given parameters
    Support for:
        * pkl file: Key 'Y' is used to access the labels
        * txt file: sparse libsvm format with header
        * npz file: numpy's sparse format

    Args:
        data_dir (str): data directory
        fname (str): load data from this file
        Y (Optional[spmatrix], optional): preloaded matrix. Defaults to None.
        normalize (bool, optional): normalize. Defaults to False.
            * Normalize the labels or not
            * Useful in case of non binary labels
        _type (str, optional): type of labels. Defaults to 'sparse'.
            * sparse: return just the indices
            * dense: return all the labels
    Raises:
        NotImplementedError: Error in case of wrong arguments

    Returns:
        LabelsBase: return the label object
    """
    if fname is None and Y is None:  # No labels are provided
        return LabelsBase(data_dir, fname, Y)
    else:
        if _type == 'sparse':
            return SparseLabels(data_dir, fname, Y, normalize)
        elif _type == 'dense':
            return DenseLabels(data_dir, fname, Y, normalize)
        else:
            raise NotImplementedError("Unknown label type")
