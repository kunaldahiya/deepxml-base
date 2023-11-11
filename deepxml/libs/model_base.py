from typing import Optional, Any
from torch import Tensor
from torch.nn import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from logging import INFO, Logger


import sys
import os
import logging
from .tracking import Tracking
import torch.utils.data
from tqdm import tqdm


class ModelBase(object):
    """Base class for Deep extreme multi-label learning

    Args:
        net (torch.nn.Module): network object
        criterion (_Loss): object to compute loss
        optimizer (Optimizer): Optimizer
        schedular (LRScheduler): learning rate schedular
        model_dir (str): directory to save model etc.
        result_dir (str): directory to save results etc.
    """
    def __init__(self,
                 net: torch.nn.Module,
                 criterion: _Loss,
                 optimizer: Optimizer,
                 schedular: LRScheduler,
                 model_dir: str,
                 result_dir: str,
                 *args: Optional[Any],
                 **kwargs: Optional[Any]) -> None:
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.schedular = schedular
        self.current_epoch = 0
        self.last_saved_epoch = -1
        self.model_dir = model_dir
        self.result_dir = result_dir
        self.last_epoch = 0
        self.model_fname = "model"
        self.logger = self.get_logger(name=self.model_fname)
        self.tracking = Tracking()
        self.scaler = None

    def setup_amp(self, use_amp: bool) -> None:
        """Set up scaler for mixed precision training

        Args:
            use_amp (bool): use automatic mixed precision or not
        """
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def get_logger(self, name: str='DeepXML', level: int=INFO) -> Logger:
        """Get logger object

        Args:
            name (str, optional): name for logging. Defaults to 'DeepXML'.
            level (int, optional): level in logger. Defaults to INFO.

        Returns:
            Logger: logger object
        """
        logger = logging.getLogger(name)
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.propagate = False
        logging.Formatter(fmt='%(levelname)s:%(message)s')
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.setLevel(level=level)
        return logger

    def _compute_loss(self, _pred: Tensor, batch_data: dict) -> Tensor:
        """Compute loss

        Args:
            pred (Tensor): predictions from the network
            batch_data (dict): dict containing (local) ground truth

        Returns:
            Tensor: computed loss
        """
        _true = batch_data['Y'].to(_pred.get_device())
        return self.criterion(_pred, _true)

    def _step_amp(self,
              data_loader: torch.utils.data.DataLoader,
              precomputed_intermediate: bool=False) -> float:
        """Training step (one pass over dataset) for amp training

        Args:
            data_loader (torch.utils.data.DataLoader): data loader over train dataset
            precomputed_intermediate (bool, optional): available already?. Defaults to False.
            if precomputed intermediate features are already available
            * avoid recomputation of intermediate features

        Returns:
            float: mean loss of all instances
        """
        self.net.train()
        torch.set_grad_enabled(True)
        mean_loss = 0
        pbar = tqdm(data_loader)
        for batch_data in pbar:
            self.optimizer.zero_grad()
            batch_size = batch_data['batch_size']
            with torch.cuda.amp.autocast():
                out_ans = self.net.forward(batch_data, precomputed_intermediate)
                loss = self._compute_loss(out_ans, batch_data)
            mean_loss += loss.item()*batch_size
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.schedular.step()
            pbar.set_description(
                f"loss: {loss.item():.5f}")
            del batch_data
        return mean_loss / data_loader.dataset.num_instances

    def _step(self,
              data_loader: torch.utils.data.DataLoader,
              precomputed_intermediate: bool=False) -> float:
        """Training step (one pass over dataset)

        Args:
            data_loader (torch.utils.data.DataLoader): data loader over train dataset
            precomputed_intermediate (bool, optional): available already?. Defaults to False.
            if precomputed intermediate features are already available
            * avoid recomputation of intermediate features

        Returns:
            float: mean loss of all instances
        """
        self.net.train()
        torch.set_grad_enabled(True)
        mean_loss = 0
        pbar = tqdm(data_loader)
        for batch_data in pbar:
            self.optimizer.zero_grad()
            batch_size = batch_data['batch_size']
            out_ans = self.net.forward(batch_data, precomputed_intermediate)
            loss = self._compute_loss(out_ans, batch_data)
            mean_loss += loss.item()*batch_size
            loss.backward()
            self.optimizer.step()
            self.schedular.step()
            pbar.set_description(
                f"loss: {loss.item():.5f}")
            del batch_data
        return mean_loss / data_loader.dataset.num_instances

    def save(self, model_dir: str, fname: str, *args: Any) -> None:
        """Save model on disk
        * uses suffix: _network.pkl for network

        Args:
            model_dir (str): save model into this directory
            fname (str): save model with this file name
        """
        fname = os.path.join(
            model_dir, fname+'_network.pkl')
        self.logger.info("Saving model at: {}".format(fname))
        state_dict = self.net.state_dict()
        torch.save(state_dict, fname)

    def load(self, model_dir: str, fname: str, *args: Any) -> None:
        """Load model from disk
        * uses suffix: _network.pkl for network

        Args:
            model_dir (str): save model into this directory
            fname (str): save model with this file name
        """
        fname_net = fname+'_network.pkl'
        state_dict = torch.load(
            os.path.join(model_dir, model_dir, fname_net))
        self.net.load_state_dict(state_dict)

    @property
    def model_size(self) -> float:
        """Returns model size

        Returns:
            float: model size in mb
        """
        return self.net.model_size
    