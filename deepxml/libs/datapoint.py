from typing import Optional, Union, Tuple
from numpy import ndarray
from torch import Tensor
from dataclasses import dataclass


@dataclass
class DataPoint:
    """
    DataPoint class represents a single data point in a dataset.

    Attributes:
        index (int): The index of the data point.
        x (torch.Tensor): The input features of the data point.
        y (Optional[Union[ndarray, Tuple]]): The target labels of the data point. Default is None.
        yf (Optional[Union[Wndarray, Tuple]]): Additional target labels or features. Default is None.
    """
    index: int
    x: Tensor
    y: Optional[Union[ndarray, Tuple]]=None
    y_neg: Optional[Union[ndarray, Tuple]]=None
    yf: Optional[Union[ndarray, Tuple]]=None
    yf_neg: Optional[Union[ndarray, Tuple]]=None
