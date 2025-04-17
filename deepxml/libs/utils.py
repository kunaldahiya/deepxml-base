from typing import Union
from scipy.sparse import spmatrix
from numpy import ndarray

import math
import numpy as np
from contextlib import contextmanager
from scipy.sparse import save_npz


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


def save_predictions(pred: spmatrix, fname: str) -> None:
    save_npz(fname, pred.tocsr())


def epochs_to_iterations(n: int, n_epochs: int, bsz: int) -> int:
    """A helper function to convert between epoch and iterations or steps
    * Useful for optimizer
    
    Args:
        n (int): number of data points
        n_epochs (int): number of epochs
        bsz (int): batch size

    Returns:
        int: number of iterations or steps
    """
    return n_epochs * math.ceil(n//bsz)


@contextmanager
def evaluating(net):
    """
    A context manager to temporarily set the model to evaluation mode.
    
    It saves the current training state of the model, switches to eval mode,
    and then restores the original state after the block is executed.
    """
    org_mode = net.training  # Save the current mode (True if training, False if eval)
    net.eval()  # Set to eval mode
    try:
        yield net
    finally:
        # Restore the model's original mode
        if org_mode:
            net.train()
