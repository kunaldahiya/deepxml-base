from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F


class _Loss(torch.nn.Module):
    """Base class for loss functions
      * Standard loss functions can also be used
      * Support for custom reduction 
          and masking which can be helpful in XC setup 
    
    Args:
        reduction (str, optional): reduction for loss. Defaults to 'mean'.
          * 'none': no reduction will be applied
          * 'mean' or 'sum': mean or sum of loss terms
          * 'custom': sum across labels and mean across data points
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


def _convert_labels_for_svm(y: Tensor) -> Tensor:
    """
        Convert labels from {0, 1} to {-1, 1}
    """
    return 2.*y - 1.0


class HingeLoss(_Loss):
    """Hinge loss
    * it'll automatically convert target to +1/-1 as required by hinge loss

    Args:
        margin (float, optional): the margin in hinge loss. Defaults to 1.0.
        reduction (str, optional): reduction for loss. Defaults to 'mean'.
          * 'none': no reduction will be applied
          * 'mean' or 'sum': mean or sum of loss terms
          * 'custom': sum across labels and mean across data points
        pad_ind (Optional[int], optional): padding index. Defaults to None.
          * ignore loss at this index (useful in 1-vs-all setting)
    """
    def __init__(
            self,
            margin: float=1.0,
            reduction: str='mean',
            pad_ind: Optional[int]=None) -> None:
        super(HingeLoss, self).__init__(reduction, pad_ind)
        self.margin = margin

    def forward(
            self,
            input: Tensor,
            target: Tensor,
            mask: Optional[Tensor]=None) -> Tensor:
        """Forward pass

        Args:
            input (Tensor): real number pred matrix of 
              size: batch_size x output_size
            target (Tensor): ground truth tensor
              0/1 ground truth matrix of size: batch_size x output_size
              * it'll automatically convert to +1/-1 as required by hinge loss
            mask: torch.BoolTensor or None, optional (default=None)
              ignore entries [won't contribute to loss] where mask value is zero

        Returns:
            Tensor: computed loss
              dimension is defined based on reduction
        """
        loss = F.relu(self.margin - _convert_labels_for_svm(target)*input)
        loss = self._mask_at_pad(loss)
        loss = self._mask(loss, mask)
        return self._reduce(loss)


class SquaredHingeLoss(_Loss):
    """Squared Hinge loss
    * it'll automatically convert target to +1/-1 as required by hinge loss

    Args:
        margin (float, optional): the margin in hinge loss. Defaults to 1.0.
        reduction (str, optional): reduction for loss. Defaults to 'mean'.
          * 'none': no reduction will be applied
          * 'mean' or 'sum': mean or sum of loss terms
          * 'custom': sum across labels and mean across data points
        pad_ind (Optional[int], optional): padding index. Defaults to None.
          * ignore loss at this index (useful in 1-vs-all setting)
    """

    def __init__(
            self,
            margin: float=1.0,
            reduction: str='mean',
            pad_ind: Optional[int]=None) -> None:
        super(SquaredHingeLoss, self).__init__(reduction, pad_ind)
        self.margin = margin

    def forward(
            self,
            input: Tensor,
            target: Tensor,
            mask: Optional[Tensor]=None) -> Tensor:
        """Forward pass

        Args:
            input (Tensor): real number pred matrix of 
              size: batch_size x output_size
            target (Tensor): ground truth tensor
              0/1 ground truth matrix of size: batch_size x output_size
              * it'll automatically convert to +1/-1 as required by hinge loss
            mask: torch.BoolTensor or None, optional (default=None)
              ignore entries [won't contribute to loss] where mask value is zero

        Returns:
            Tensor: computed loss
              dimension is defined based on reduction
        """
        loss = F.relu(self.margin - _convert_labels_for_svm(target)*input)
        loss = loss**2
        loss = self._mask_at_pad(loss)
        loss = self._mask(loss, mask)
        return self._reduce(loss)


class BCEWithLogitsLoss(_Loss):
    """BCE loss (expects logits; numercial stable)
    This loss combines a `Sigmoid` layer and the `BCELoss` in one single
    class. This version is more numerically stable than using a plain `Sigmoid`
    followed by a `BCELoss` as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.

    Args:
        weight: (Tensor, optional), optional. Defaults to None.
          a manual rescaling weight given to the loss of each batch element.
          If given, has to be a Tensor of size batch_size
        reduction (str, optional): reduction for loss. Defaults to 'mean'.
          * 'none': no reduction will be applied
          * 'mean' or 'sum': mean or sum of loss terms
          * 'custom': sum across labels and mean across data points
        pos_weight: (Optional[Tensor], optional): weight of positives. Defaults to None.
          it must be a vector with length equal to the number of classes.
        pad_ind (Optional[int], optional): padding index. Defaults to None.
          * ignore loss at this index (useful in 1-vs-all setting)
    """
    __constants__ = ['weight', 'pos_weight', 'reduction']

    def __init__(self, weight=None, reduction='mean',
                 pos_weight=None, pad_ind=None):
        super(BCEWithLogitsLoss, self).__init__(reduction, pad_ind)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target, mask=None):
        """Forward pass

        Args:
            input (Tensor): real number pred matrix of 
              size: batch_size x output_size
            target (Tensor): ground truth tensor
              0/1 ground truth matrix of size: batch_size x output_size
              * it'll automatically convert to +1/-1 as required by hinge loss
            mask: torch.BoolTensor or None, optional (default=None)
              ignore entries [won't contribute to loss] where mask value is zero

        Returns:
            Tensor: computed loss
              dimension is defined based on reduction
        """
        loss = F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction='none')
        loss = self._mask_at_pad(loss)
        loss = self._mask(loss, mask)
        return self._reduce(loss)
