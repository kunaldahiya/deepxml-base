from typing import Optional, Any, Tuple, Union, Callable
from scipy.sparse import spmatrix
from torch import Tensor
from .loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from logging import INFO, Logger
from .dataset_base import DatasetBase


import sys
import os
import logging
import time
import torch.utils.data
from .tracking import Tracking
from torch.utils.data import DataLoader
from tqdm import tqdm
from xclib.utils.matrix import SMatrix
from .dataset import construct_dataset
from .collator import collate
from .evaluater import Evaluater


class ModelBase(object):
    """Base class for Deep extreme multi-label learning
    """
    def __init__(self,
                 net: torch.nn.Module,
                 criterion: _Loss,
                 optimizer: Optimizer,
                 schedular: LRScheduler,
                 evaluater: Evaluater,
                 model_dir: str,
                 result_dir: str,
                 *args: Optional[Any],
                 **kwargs: Optional[Any]) -> None:
        """ 
        Args:
            net (torch.nn.Module): network object
            criterion (_Loss): object to compute loss
            optimizer (Optimizer): Optimizer
            schedular (LRScheduler): learning rate schedular
            evaluater (Evaluater): to evaluate
            model_dir (str): directory to save model etc.
            result_dir (str): directory to save results etc.
        """
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.schedular = schedular
        self.evaluater = evaluater
        self.current_epoch = 0
        self.last_saved_epoch = -1
        self.model_dir = model_dir
        self.result_dir = result_dir
        self.last_epoch = 0
        self.model_fname = "model"
        self.logger = self.get_logger(name=self.model_fname)
        self.tracking = Tracking()
        self.device = self.net.device
        self.scaler = None

    def setup_amp(self, use_amp: bool) -> None:
        """Set up scaler for mixed precision training

        Args:
            use_amp (bool): use automatic mixed precision or not
        """
        if use_amp:
            self.scaler = torch.amp.GradScaler(self.device)

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

    def _create_dataset(
            self,
            data_dir,
            fname,
            data=None,
            mode='train',
            normalize_features=True,
            normalize_labels=True,
            feature_type='sparse',
            max_len=-1,
            **kwargs) -> DatasetBase:
            return construct_dataset(
                data_dir,
                fname,
                data=data,
                mode=mode,
                max_len=max_len,
                feature_type=feature_type,
                normalize_features=normalize_features,
                normalize_labels=normalize_labels,
                **kwargs
            )

    def _create_data_loader(
        self,
        dataset: DatasetBase,
        prefetch_factor: int=5,
        batch_size: int=128,
        feature_type: str='sparse',
        sampling_type: str='brute',
        num_workers: int=4,
        shuffle: bool=False, 
        **kwargs
    ) -> DataLoader:
        """Create data loader for given dataset

        Args:
            dataset (DatasetBase): Dataset object
            prefetch_factor (int, optional): Defaults to 5
                used in the data loader. 
            batch_size (int, optional): Defaults to 128
                batch size in data loader.
            feature_type (str, optional): Defaults to 'sparse'
                type of features.
            sampling_type (str, optional): Defaults to 'brute'.
                sampling type (used in creating data loader)
            num_workers (int, optional): Workers in dataloader. Defaults to 4.
            shuffle (bool, optional): Shuffle batches. Defaults to False.

        Returns:
            DataLoader: Data loader
        """
        dt_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=self._create_collate_fn(
                feature_type, sampling_type))
        return dt_loader

    def _create_collate_fn(self, feature_type, sampling_type):
        return collate(feature_type, sampling_type)

    def _compute_loss(self, _pred: Tensor, batch_data: dict) -> Tensor:
        """Compute loss

        Args:
            pred (Tensor): predictions from the network
            batch_data (dict): dict containing (local) ground truth

        Returns:
            Tensor: computed loss
        """
        _true = batch_data['Y'].to(self.device)
        return self.criterion(_pred, _true)

    def _step_amp(self,
              data_loader: DataLoader,
              precomputed_intermediate: bool=False) -> float:
        """Training step (one pass over dataset) for amp training

        Args:
            data_loader (DataLoader): data loader over train dataset
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
            with torch.amp.autocast(self.device):
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
              data_loader: DataLoader,
              precomputed_intermediate: bool=False) -> float:
        """Training step (one pass over dataset)

        Args:
            data_loader (DataLoader): data loader over train dataset
            precomputed_intermediate (bool, optional): available already?.
                Defaults to False.
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

    def evaluate(
            self, 
            _true: spmatrix, 
            _pred: spmatrix, 
            k: int=5, 
            filter_map: str=None):
        return self.evaluater(_true, _pred, k=k, filter_map=filter_map)

    def validate(
            self,
            data_loader: DataLoader,
            epoch:int=-1,
    ) -> None:
        """
        Validate for the given data loader

        Arguments
        ---------
        validation_loader: DataLoader or None
            data loader over validation dataset
        epoch (int): used in saving checkpoint
        """
        tic = time.time()
        predicted_labels, val_avg_loss = self._validate(data_loader)
        toc = time.time()
        _prec = self.evaluate(
            data_loader.dataset.labels.Y,
            predicted_labels,
            filter_map=getattr(data_loader.dataset, 'label_filter', None))
        self.logger.info("Model saved after epoch: {}".format(epoch))
        self.save_checkpoint(self.model_dir, epoch+1)
        self.tracking.last_saved_epoch = epoch
        self.logger.info(
            f"{_prec.summary()}, loss: {val_avg_loss:.6f}, "\
            f"time: {toc-tic:.2f} sec")

    def _fit(
        self,
        train_loader: DataLoader,
        validation_loader: Union[DataLoader, None],
        num_epochs: int=10,
        validation_interval: int=5
    ) -> None:
        """
        Train for the given data loader

        Arguments
        ---------
        train_loader: DataLoader
            data loader over train dataset
        validation_loader: DataLoader or None
            data loader over validation dataset
        num_epochs: int
            #passes over the dataset
        validate_after: int, optional, default=5
            validate after a gap of these many epochs
        precomputed_intermediate: boolean, optional, default=False
            if precomputed intermediate features are already available
            * avoid recomputation of intermediate features
        """
        for epoch in range(num_epochs):
            tic = time.time()
            avg_loss = self._step(train_loader)
            toc = time.time()
            self.logger.info(
                f"Epoch: {epoch}, loss: {avg_loss:.6f}, time: {toc-tic:.2f} sec")
            if validation_loader is not None and epoch % validation_interval == 0:
                self.validate(validation_loader, epoch)
        self.save_checkpoint(self.model_dir, epoch+1)

    @torch.no_grad()
    def _validate(
            self, 
            data_loader: DataLoader, 
            k: int=10) -> Tuple[spmatrix, float]:
        """predict for the given data loader

        Args:
            data_loader (DataLoader): data loader 
                over validation dataset
            k (int, optional): consider top k predictions. 
                Defaults to 10.

        Returns:
            tuple (spmatrix, float) 
                - predictions for the given dataset
                - mean loss over the validation dataset
        """
        self.net.eval()
        top_k = min(k, data_loader.dataset.num_labels)
        mean_loss = 0
        predicted_labels = SMatrix(
            n_rows=data_loader.dataset.num_instances,
            n_cols=data_loader.dataset.num_labels,
            nnz=top_k)
        count = 0
        for batch_data in tqdm(data_loader):
            batch_size = batch_data['batch_size']
            out_ans = self.net.forward(batch_data)
            loss = self._compute_loss(out_ans, batch_data)
            mean_loss += loss.item()*batch_size
            vals, ind = torch.topk(out_ans, k=top_k, dim=-1, sorted=False)
            predicted_labels.update_block(
                count, ind.cpu().numpy(), vals.cpu().numpy())
            count += batch_size
            del batch_data
        return predicted_labels.data(), \
            mean_loss / data_loader.dataset.num_instances

    def fit(
        self,
        data_dir: str,
        dataset: str,
        trn_fname: str,
        val_fname: str,
        trn_data: dict=None,
        val_data: dict=None,
        num_epochs: int=10,
        batch_size: int=128,
        num_workers: int=4,
        shuffle: bool=True,
        feature_type: str='dense',
        normalize_features=True,
        normalize_labels=False,
        validate_interval=5,
        surrogate_mapping=None, **kwargs
    ) -> None:
        """Train the model on the basis of given data and parameters

        Args:
            data_dir (str): load data from this directory when data is None
            dataset (str): dataset name
            trn_fname (str): file names for training data
            val_fname (str): file names for validation data
            trn_data (dict, optional): loaded train data. Defaults to None.
            val_data (dict, optional): loaded val data. Defaults to None.
            num_epochs (int, optional): number of epochs. Defaults to 10.
            batch_size (int, optional): batch size. Defaults to 128.
            num_workers (int, optional): workers in data loader. Defaults to 4.
            shuffle (bool, optional): shuffle while batching. Defaults to False.
            normalize_features (bool, optional): normalize features. Defaults to True.
            normalize_labels (bool, optional): normalize labels. Defaults to False.
            validate_interval (int, optional): validate after these many epochs. 
                Defaults to 5.
            surrogate_mapping (_type_, optional): _description_. Defaults to None.
        """
        # Reset the logger to dump in train log file
        self.logger.addHandler(
            logging.FileHandler(os.path.join(self.result_dir, 'log_train.txt'))) 
        self.logger.info("Loading training data.")
        train_dataset = self._create_dataset(
            os.path.join(data_dir, dataset),
            trn_fname,
            data=trn_data,
            mode='train',
            feature_type=feature_type,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            surrogate_mapping=surrogate_mapping)
        train_loader = self._create_data_loader(
            train_dataset,
            batch_size=batch_size,
            feature_type=feature_type,
            num_workers=num_workers,
            shuffle=shuffle)
        self.logger.info("Loading validation data.")
        validation_loader = None
        if validate_interval < num_epochs:
            validation_dataset = self._create_dataset(
                os.path.join(data_dir, dataset),
                val_fname,
                data=val_data,
                mode='predict',
                feature_type=feature_type,
                normalize_features=normalize_features,
                normalize_labels=normalize_labels,
                surrogate_mapping=surrogate_mapping)
            validation_loader = self._create_data_loader(
                validation_dataset,
                feature_type=feature_type,
                batch_size=batch_size,
                num_workers=num_workers)
        self._fit(train_loader, validation_loader,
                  num_epochs, validate_interval)

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

    def save_checkpoint(self, model_dir, epoch, do_purge=True):
        """
        Save checkpoint on disk
        * save network, optimizer and loss
        * filename: checkpoint_net_epoch.pkl for network

        Arguments:
        ---------
        model_dir: str
            save checkpoint into this directory
        epoch: int
            checkpoint after this epoch (used in file name)
        do_purge: boolean, optional, default=True
            delete old checkpoints beyond a point
        """
        checkpoint = {
            'epoch': epoch,
            'criterion': self.criterion.state_dict(),
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(
            checkpoint, 
            os.path.join(model_dir, f'checkpoint_{epoch}.pkl'))

    def load_checkpoint(self, model_dir, fname, epoch):
        """
        Load checkpoint from disk
        * load network, optimizer and loss
        * filename: checkpoint_net_epoch.pkl for network

        Arguments:
        ---------
        model_dir: str
            load checkpoint into this directory
        epoch: int
            checkpoint after this epoch (used in file name)
        """
        fname = os.path.join(model_dir, f'checkpoint_{epoch}.pkl')
        checkpoint = torch.load(open(fname, 'rb'))
        self.net.load_state_dict(checkpoint['net'])
        self.criterion.load_state_dict(checkpoint['criterion'])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
