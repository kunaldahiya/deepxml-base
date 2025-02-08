from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F


class _Loss(torch.nn.Module):
    """Base class for loss functions
      * Standard loss functions can also be used
      * Support for custom reduction 
          and masking which can be helpful in XC setup     
    """
    def __init__(
            self,
            reduction: str = 'mean',
            pad_ind: int = None) -> None:
        """
        Args:
            reduction (str, optional): how to reduce the batch. Defaults to 'mean'.
              - 'none': no reduction will be applied
              - 'mean' or 'sum': mean or sum of loss terms
              - 'custom': sum across labels and mean across data points
            pad_ind (Optional[int], optional): padding index. Defaults to None.
              - ignore loss at this index (useful in 1-vs-all setting)
        """
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

    def _mask(self, loss: Tensor, mask: Tensor=None) -> Tensor:
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
    """
    def __init__(
            self,
            margin: float = 1.0,
            reduction: str = 'mean',
            pad_ind: int = None) -> None:
        """
        Args:
            margin (float, optional): the margin in hinge loss. Defaults to 1.0.
            reduction (str, optional): how to reduce the batch. Defaults to 'mean'.
              - 'none': no reduction will be applied
              - 'mean' or 'sum': mean or sum of loss terms
              - 'custom': sum across labels and mean across data points
            pad_ind (Optional[int], optional): padding index. Defaults to None.
              - ignore loss at this index (useful in 1-vs-all setting)
        """
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
    """

    def __init__(
            self,
            margin: float = 1.0,
            reduction: str = 'mean',
            pad_ind: int = None) -> None:
        """
        Args:
            margin (float, optional): the margin in hinge loss. Defaults to 1.0.
            reduction (str, optional): how to reduce the batch. Defaults to 'mean'.
              - 'none': no reduction will be applied
              - 'mean' or 'sum': mean or sum of loss terms
              - 'custom': sum across labels and mean across data points
            pad_ind (Optional[int], optional): padding index. Defaults to None.
              - ignore loss at this index (useful in 1-vs-all setting)
        """
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
    """BCE loss (expects logits; numerically stable)
    """
    __constants__ = ['weight', 'pos_weight', 'reduction']

    def __init__(
            self,
            weight: torch.Tensor = None,
            reduction: str = 'mean',
            pos_weight: Tensor = None,
            pad_ind: int = None):
        """
        Args:
            weight (torch.Tensor, optional): Defaults to None.
              a manual rescaling weight given to the loss of each batch element.
              If given, has to be a Tensor of size batch_size
            reduction (str, optional): how to reduce the batch. Defaults to 'mean'.
              - 'none': no reduction will be applied
              - 'mean' or 'sum': mean or sum of loss terms
              - 'custom': sum across labels and mean across data points
            pos_weight (Tensor, optional): weight of positives. Defaults to None.
              it must be a vector with length equal to the number of classes.
            pad_ind (int, optional): padding index. Defaults to None.
              - ignore loss at this index (useful in 1-vs-all setting)
        """
        super(BCEWithLogitsLoss, self).__init__(reduction, pad_ind)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input: Tensor, target: Tensor, mask: Tensor=None) -> Tensor:
        """
        Args:
            output (Tensor): similarity b/w label and document
              real number pred matrix of size: batch_size x output_size
            target (Tensor): 0/1 ground truth matrix of 
              size: batch_size x output_size
            mask (Tensor, optional): A boolean mask (ignore certain entries)

        Returns:
            torch.Tensor: loss for the given data
        """
        loss = F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction='none')
        loss = self._mask_at_pad(loss)
        loss = self._mask(loss, mask)
        return self._reduce(loss)


class TripletMarginLossOHNM(_Loss):
    """ Triplet Margin Loss with Online Hard Negative Mining
    * Applies loss using the hardest negative in the mini-batch
    * Assumes diagonal entries are ground truth
    """

    def __init__(
            self,
            margin: float = 1.0,
            eps: float = 1.0e-6,
            reduction: str = 'mean',
            num_negatives: int = 10,
            num_violators: bool = False,
            tau: float = 0.1):
        """
        Args:
            margin (float, optional): Margin in triplet loss. Defaults to 1.0
            eps (float, optional): for numerical safety. Defaults to 1.0e-6.
            reduction (str, optional): how to reduce the batch. Defaults to 'mean'
              - 'none': no reduction will be applied
              - 'mean' or 'sum': mean or sum of loss terms
              - 'custom': sum over the labels and mean acorss data-points
            num_negatives (int, optional): #negatives per data point. Defaults to 10
            num_violators (bool, optional): Defaults to False.
              number of labels violating the margin.
            tau (float, optional): temprature in similarity. Defaults to 0.1.
              rescale the logits as per the temprature (tau)
        """
        super(TripletMarginLossOHNM, self).__init__(reduction)
        self.mx_lim = 100
        self.mn_lim = -100
        self.tau = tau
        self._eps = eps
        self.margin = margin
        self.reduction = reduction
        self.num_negatives = num_negatives
        self.num_violators = num_violators
        self.recale = tau != 1.0

    def forward(
            self, 
            output: torch.Tensor, 
            target: torch.Tensor, *args) -> torch.Tensor:
        """
        Args:
            output (torch.Tensor): cosine similarity b/w label and document
              real number pred matrix of size: batch_size x output_size
            target (torch.Tensor): 0/1 ground truth matrix of 
              size: batch_size x output_size

        Returns:
            torch.Tensor: loss for the given data
        """
        with torch.no_grad():
            indices = torch.multinomial(
                target, 
                num_samples=1, 
                replacement=False)

        sim_p = output.gather(1, indices.view(-1, 1))

        similarities = torch.where(
            target == 0, 
            output, 
            torch.full_like(output, self.mn_lim))

        _, indices = torch.topk(
            similarities, 
            largest=True, 
            dim=1, 
            k=self.num_negatives)

        sim_n = output.gather(1, indices)

        loss = torch.max(
            torch.zeros_like(sim_p), 
            sim_n - sim_p + self.margin)

        if self.recale:
            sim_n[loss == 0] = self.mn_lim
            prob = torch.softmax(sim_n/self.tau, dim=1)
            loss = loss * prob
        
        reduced_loss = self._reduce(loss)
        if self.num_violators:
            nnz = torch.sum((loss > 0), axis=1).float().mean()
            return reduced_loss, nnz
        else:
            return reduced_loss
