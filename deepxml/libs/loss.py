from typing import Optional

import torch
from torch import Tensor



class _Loss(torch.nn.Module):
    """Base class for loss functions
        * Standard loss functions can also be used
        * Support for custom reduction 
          and masking which can be helpful in XC setup 
    
    Args:
        reduction (str, optional): reduce using this function. Defaults to 'mean'.
          * 'mean' or 'none' or 'sum' or 'custom'
        pad_ind (Optional[int], optional): padding index. Defaults to None.
          * ignore loss at this index (useful in 1-vs-all setting)
    """
    def __init__(self, reduction: str='mean', pad_ind: Optional[int]=None) -> None:
        super(_Loss, self).__init__()
        self.reduction = reduction
        self.pad_ind = pad_ind

    def _reduce(self, loss: Tensor) -> Tensor:
        """Reduce the loss as per reduction function

        Args:
            loss (Tensor): loss tensor 
            * usually shape (batch size, #labels)

        Returns:
            Tensor: reduced loss
        """
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'custom':
            return loss.sum(dim=1).mean()
        else:
            return loss.sum()

    def _mask_at_pad(self, loss: Tensor) -> Tensor:
        """Mask the loss at padding index, i.e., make it zero

        Args:
            loss (Tensor): loss tensor 
              * usually shape (batch size, #labels)

        Returns:
            Tensor: modified loss tensor
        """
        if self.pad_ind is not None:
            loss[:, self.pad_ind] = 0.0
        return loss

    def _mask(self, loss: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        """Mask the loss at padding index, i.e., make it zero

        Args:
            loss (Tensor): loss tensor 
              * usually shape (batch size, #labels)
            mask (Optional[Tensor], optional): mask tensor. Defaults to None.
              * Mask should be a boolean array with 1 where loss needs
              to be considered.
              * it'll make it zero where value is 0

        Returns:
            Tensor: modified loss tensor
        """
        if mask is not None:
            loss = loss.masked_fill(~mask, 0.0)
        return loss
