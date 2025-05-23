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
import time
import logging
import numpy as np
from tqdm import tqdm
import torch.utils.data
from .timer import Timer
from .tracking import Tracking
from torch.utils.data import DataLoader
from xclib.utils.matrix import SMatrix
from .dataset import construct_dataset
from .dataset_factory import DatasetFactory
from .collator import collate
from .evaluater import Evaluater


class PipelineBase(object):
    """Base class for Deep extreme multi-label learning
    """
    def __init__(self,
                 net: torch.nn.Module,
                 model_dir: str,
                 result_dir: str,
                 criterion: _Loss=None,
                 optimizer: Optimizer=None,
                 schedular: LRScheduler=None,
                 evaluater: Evaluater=None,
                 use_amp: bool = True,
                 *args: Optional[Any],
                 **kwargs: Optional[Any]) -> None:
        """
        Args:
            net (torch.nn.Module): network object
            model_dir (str): directory to save model etc.
            result_dir (str): directory to save results etc.
            criterion (_Loss, optional): to compute loss. Defaults to None.
            optimizer (Optimizer, optional): Optimizer. Defaults to None.
            schedular (LRScheduler, optional): learning rate schedular. Defaults to None.
            evaluater (Evaluater, optional): to evaluate. Defaults to None.
            use_amp (bool, optional): mixed precision. Defaults to True.
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
        self.device = str(self.net.device)
        self.setup_amp(use_amp)
        self.setup_tracking()

    def setup_tracking(self):
        "Initialize logger, tracker and timers"
        self.train_timer = Timer()
        self.predict_timer = Timer()
        self.val_timer = Timer()
        self.tracking = Tracking()
        self.logger = self.get_logger(name=self.model_fname)

    def setup_amp(self, use_amp: bool) -> None:
        """Set up scaler for mixed precision training

        Args:
            use_amp (bool): use automatic mixed precision or not
        """
        self.use_amp = use_amp
        if use_amp:
            self.scaler = torch.amp.GradScaler(self.device)
        else:
            self.scaler = None

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

    def _dataset_factory(self):
        """This function allows the child method to inherit the class
        to define its own datasets. They can just redefine the class 
        to load from their local code. Otherwise more code change is required

        Returns:
            dict: A dataset factory that can return the Dataset class based 
            on the key (sampling_t)
        """
        return DatasetFactory 

    def _create_dataset(
            self,
            data_dir,
            fname,
            data=None,
            mode='train',
            normalize_features=True,
            normalize_labels=True,
            feature_t='sequential',
            max_len=-1,
            **kwargs) -> DatasetBase:
            return construct_dataset(
                data_dir,
                fname,
                data=data,
                mode=mode,
                max_len=max_len,
                feature_t=feature_t,
                normalize_features=normalize_features,
                normalize_labels=normalize_labels,
                dataset_factory=self._dataset_factory(),
                **kwargs
            )

    def _create_data_loader(
        self,
        dataset: DatasetBase,
        prefetch_factor: int=5,
        batch_size: int=128,
        feature_t: str='sequential',
        op_feature_t: str=None,
        sampling_t: str='brute',
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
            feature_t (str, optional): Defaults to 'sequential'
                type of features.
            sampling_t (str, optional): Defaults to 'brute'.
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
                feature_t, sampling_t, op_feature_t))
        return dt_loader

    def _create_collate_fn(self, feature_t, sampling_t, op_feature_t):
        return collate(feature_t, sampling_t, op_feature_t)

    def _compute_loss(self, _pred: Tensor, batch: dict) -> Tensor:
        """Compute loss

        Args:
            pred (Tensor): predictions from the network
            batch (dict): dict containing (local) ground truth

        Returns:
            Tensor: computed loss
        """
        _true = batch['Y'].to(self.device)
        return self.criterion(_pred, _true)

    def _step_amp(self,
              batch: dict,
              *args, **kwargs) -> float:
        """Training step (process one batch)

        Args:
            batch (dict): batch data
            precomputed_intermediate (bool, optional): available already?.
                Defaults to False.
                if precomputed intermediate features are already available
                * avoid recomputation of intermediate features

        Returns:
            float: loss value as float
        """
        self.optimizer.zero_grad()
        with torch.amp.autocast(self.device):
            out = self.net.forward(batch, *args, **kwargs)
            loss = self._compute_loss(out, batch)
        _loss = loss.item()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.schedular.step()
        return _loss

    def _step(self,
              batch: dict,
              precomputed_intermediate: bool=False) -> float:
        """Training step (process one batch)

        Args:
            batch (dict): batch data
            precomputed_intermediate (bool, optional): available already?.
                Defaults to False.
                if precomputed intermediate features are already available
                * avoid recomputation of intermediate features

        Returns:
            float: loss value as float
        """
        self.optimizer.zero_grad()
        out = self.net.forward(batch, precomputed_intermediate)
        loss = self._compute_loss(out, batch)
        _loss = loss.item()
        loss.backward()
        self.optimizer.step()
        self.schedular.step()
        return _loss

    def _epoch(self,
              data_loader: DataLoader,
              *args, **kwargs) -> float:
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
        for batch in (pbar := tqdm(data_loader)):
            if self.use_amp:
                _loss = self._step_amp(batch, *args, **kwargs)
            else:
                _loss = self._step(batch, *args, **kwargs)
            mean_loss += _loss * batch['batch_size']
            pbar.set_description(f"loss: {_loss:.5f}")
            del batch
        return mean_loss / data_loader.dataset.num_instances

    def evaluate(
            self, 
            _true: spmatrix, 
            _pred: spmatrix, 
            k: int=5, 
            filter_map: str=None):
        return self.evaluater(_true, _pred, k=k, filter_map=filter_map)

    @torch.no_grad()
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
        self.save_checkpoint(epoch+1)
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
        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            self.train_timer.tic()
            avg_loss = self._step(train_loader)
            self.train_timer.toc()
            self.logger.info(
                f"Epoch: {epoch}, loss: {avg_loss:.6f}, time: {self.train_timer.elapsed_time:.2f} sec")
            if validation_loader is not None and epoch % validation_interval == 0:
                self.val_timer.tic()
                self.validate(validation_loader, epoch)
                self.val_timer.toc()
        self.save_checkpoint(epoch+1)

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
        for batch in tqdm(data_loader, desc="Validating"):
            bsz = batch['batch_size']
            out_ans = self.net.forward(batch)
            loss = self._compute_loss(out_ans, batch)
            mean_loss += loss.item()*bsz
            vals, ind = torch.topk(out_ans, k=top_k, dim=-1, sorted=False)
            predicted_labels.update_block(
                count, ind.cpu().numpy(), vals.cpu().numpy())
            count += bsz
            del batch
        return predicted_labels.data(), \
            mean_loss / data_loader.dataset.num_instances

    def fit(
        self,
        data_dir: str,
        trn_fname: str,
        val_fname: str,
        trn_data: dict=None,
        val_data: dict=None,
        num_epochs: int=10,
        batch_size: int=128,
        num_workers: int=4,
        shuffle: bool=True,
        feature_t: str='dense',
        normalize_features=True,
        normalize_labels=False,
        validate_interval=5,
        **kwargs
    ) -> None:
        """Train the model on the basis of given data and parameters

        Args:
            data_dir (str): load data from this directory when data is None
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
        """
        # Reset the logger to dump in train log file
        self.logger.addHandler(
            logging.FileHandler(os.path.join(self.result_dir, 'log_train.txt'))) 
        self.logger.info("Loading training data.")
        train_dataset = self._create_dataset(
            data_dir,
            trn_fname,
            data=trn_data,
            mode='train',
            feature_t=feature_t,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels)
        train_loader = self._create_data_loader(
            train_dataset,
            batch_size=batch_size,
            feature_t=feature_t,
            num_workers=num_workers,
            shuffle=shuffle)
        self.logger.info("Loading validation data.")
        validation_loader = None
        if validate_interval > 0 and validate_interval < num_epochs:
            validation_dataset = self._create_dataset(
                data_dir,
                val_fname,
                data=val_data,
                mode='predict',
                feature_t=feature_t,
                normalize_features=normalize_features,
                normalize_labels=normalize_labels)
            validation_loader = self._create_data_loader(
                validation_dataset,
                feature_t=feature_t,
                batch_size=batch_size,
                num_workers=num_workers)
        self._fit(train_loader, validation_loader,
                  num_epochs, validate_interval)
        self.logger.info(f"Train time (sec): {self.train_timer.elapsed_time}") 
        self.logger.info(f"Val time (sec): {self.val_timer.elapsed_time}") 
        self.logger.info(f"Model Size (MB): {self.model_size}") 

    @torch.no_grad()
    def _predict(
            self, 
            data_loader: DataLoader, 
            k: int=10) -> spmatrix:
        """predict for the given data loader

        Args:
            data_loader (DataLoader): data loader 
                over validation dataset
            k (int, optional): consider top k predictions. 
                Defaults to 10.

        Returns:
            spmatrix: predictions for the given dataset
        """
        self.net.eval()
        top_k = min(k, data_loader.dataset.num_labels)
        predicted_labels = SMatrix(
            n_rows=data_loader.dataset.num_instances,
            n_cols=data_loader.dataset.num_labels,
            nnz=top_k)
        count = 0
        for batch in tqdm(data_loader, desc="Predicting"):
            out = self.net.forward(batch)
            vals, ind = torch.topk(out, k=k, dim=-1, sorted=False)
            predicted_labels.update_block(
                count, ind.cpu().numpy(), vals.cpu().numpy())
            count += batch['batch_size']
            del batch
        return predicted_labels.data()

    def predict(
            self,
            data_dir: str,
            fname: str,
            data: dict=None,
            num_workers: int=4,
            batch_size: int=128,
            normalize_features: bool=True,
            normalize_labels: bool=False,
            k: int=100,
            feature_t: str='sequential'
            ) -> spmatrix: 
        """Make predictions for given file

        Args:
            data_dir (str): data directory
            fname (str): name of the file
            data (dict, optional): Preloaded data. Defaults to None.
            num_workers (int, optional): #workers. Defaults to 4.
            batch_size (int, optional): Batch size while inferring. Defaults to 128.
            normalize_features (bool, optional): Normalize features. Defaults to True.
            k (int, optional): consider top k predictions.. Defaults to 100.
            feature_t (str, optional): feature type. Defaults to 'sequential'.

        Returns:
            spmatrix: predictions for the given dataset
        """
        self.logger.addHandler(
            logging.FileHandler(os.path.join(self.result_dir, 'log_predict.txt')))
        self.logger.info("Loading test data.")
        dataset = self._create_dataset(
            data_dir,
            fname,
            data=data,
            mode='predict',
            feature_t=feature_t,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels)
        data_loader = self._create_data_loader(
            dataset,
            feature_t=feature_t,
            batch_size=batch_size,
            num_workers=num_workers)
        self.predict_timer.tic()
        predicted_labels = self._predict(data_loader, k)
        self.predict_timer.toc()
        avg_time = self.predict_timer.elapsed_time * 1000 / len(dataset)
        self.logger.info(f"Avg. inference time: {avg_time:.2f} msec")
        return predicted_labels

    @torch.no_grad()
    def _embeddings(
        self,
        data_loader: DataLoader,
        encoder: Callable = None,
        fname_out: str = None,
        _dtype='float32'
    ) -> np.ndarray:
        """Encode given data points
        * support for objects or files on disk


        Args:
            data_loader (DataLoader): DataLoader object to \
                  create batches and iterate over it
            encoder (Callable, optional): Defaults to None.
                use this function to encode given dataset
                * net.encode is used when None
            fname_out (str, optional): dump features to this file. Defaults to None.
            _dtype (str, optional): data type of output tensors. Defaults to 'float32'.

        Returns:
            np.ndarray: embeddings (as memmap or ndarray)
        """

        if encoder is None:
            self.logger.info("Using the default encoder.")
            encoder = self.net.encode
        self.net.eval()
        torch.set_grad_enabled(False)
        if fname_out is not None:  # Save to disk
            embeddings = np.memmap(
                fname_out, dtype=_dtype, mode='w+',
                shape=(len(data_loader.dataset), self.net.repr_dims))
        else:  # Keep in memory
            embeddings = np.zeros((
                len(data_loader.dataset), self.net.repr_dims),
                dtype=_dtype)
        idx = 0
        for batch in tqdm(data_loader, desc="Computing Embeddings"):
            bsz = batch['batch_size']
            out = encoder(batch['X'])
            embeddings[idx :idx+bsz, :] = out.detach().cpu().numpy()
            idx += bsz
        torch.cuda.empty_cache()
        if fname_out is not None:  # Flush all changes to disk
            embeddings.flush()
        return embeddings

    def get_embeddings(
        self,
        encoder: Callable = None,
        data_dir: str = None,
        fname: str = None,
        data: dict = None,
        batch_size: int = 1024,
        num_workers: int = 6,
        normalize: bool = False,
        fname_out: str = None,
        feature_t='sequential', 
        **kwargs
    ) -> np.ndarray:
        """Encode given data points
        * support for objects or files on disk

        Args:
            encoder (Callable, optional): Defaults to None.
                use this function to encode given dataset
                * net.encode is used when None
            data_dir (str, optional): Data directory. Defaults to None.
            fname (str, optional): data file. Defaults to None.
            data (dict, optional): pre-loaded data dictionary. Defaults to None.
            batch_size (int, optional): batch size. Defaults to 1024.
            num_workers (int, optional): number of workers. Defaults to 6.
            normalize (bool, optional): normalize features. Defaults to False.
            fname_out (str, optional): dump features to this file. Defaults to None.
            feature_t (str, optional): type of features. Defaults to 'sequential'.

        Returns:
            ndarray: numpy array containing the embeddings 
        """
        if data is None:
            assert data_dir is not None and fname is not None, \
                "valid file path is required when data is not passed"
        dataset = self._create_dataset(
            data_dir,
            fname=fname,
            data=data,
            mode="test",
            normalize_features=normalize,
            feature_t=feature_t,
            classifier_t=None,
            sampling_t=None,
            **kwargs)
        # Adjust batch size if the remainder of len(dataset) / batch_size is 1
        if len(dataset) % batch_size == 1:
            batch_size -= 1

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self._create_collate_fn(
                feature_t=feature_t,
                sampling_t=None,
                op_feature_t=None),
            shuffle=False)
        return self._embeddings(data_loader, encoder, fname_out)

    def save(self, fname: str='model', *args: Any) -> None:
        """Save model on disk
        * uses suffix: network.pt for network

        Args:
            fname (str): save model with this file name
        """
        fname = os.path.join(
            self.model_dir, fname+'.network.pt')
        self.logger.info("Saving model at: {}".format(fname))
        state_dict = self.net.state_dict()
        torch.save(state_dict, fname)

    def load(self, fname: str, *args: Any) -> None:
        """Load model from disk
        * uses suffix: network.pt for network

        Args:
            fname (str): load model with this file name
        """
        fname_net = fname+'.network.pt'
        state_dict = torch.load(
            os.path.join(self.model_dir, fname_net), weights_only=True)
        self.net.load_state_dict(state_dict)

    @property
    def model_size(self) -> float:
        """Returns model size

        Returns:
            float: model size in mb
        """
        return self.net.model_size

    def save_checkpoint(self, epoch, do_purge=True):
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
            os.path.join(self.model_dir, f'checkpoint_{epoch}.pkl'))

    def load_checkpoint(self, fname, epoch):
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
        fname = os.path.join(self.model_dir, f'checkpoint_{epoch}.pkl')
        checkpoint = torch.load(open(fname, 'rb'), weights_only=True)
        self.net.load_state_dict(checkpoint['net'])
        self.criterion.load_state_dict(checkpoint['criterion'])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
