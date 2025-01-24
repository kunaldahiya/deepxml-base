from typing import Union
from scipy.sparse import spmatrix
from numpy import ndarray

import math
import numpy as np



def compute_depth_of_tree(n: int, s: int) -> int:
    """Get depth of tree 

    Args:
        n (int): Total number of items at root node 
        s (int): Cluster size at the leaf node 

    Returns:
        int: Depth of tree
    """
    return int(math.ceil(math.log(n / s) / math.log(2)))


def get_filter_map(fname: str) -> Union[ndarray, None]:
    if fname is not None:
        return np.loadtxt(fname).astype(np.int)
    else:
        return None


def filter_predictions(pred: spmatrix, mapping: ndarray=None) -> spmatrix:
    if mapping is not None and len(mapping) > 0:
        pred[mapping[:, 0], mapping[:, 1]] = 0
        pred.eliminate_zeros()
    return pred
