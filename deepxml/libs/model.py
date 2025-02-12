from typing import Optional, Any, Tuple, Union
from numpy import ndarray
from torch import Tensor
from .loss import _Loss
from argparse import Namespace
from torch.optim import Optimizer
from scipy.sparse import spmatrix
from .dataset_base import DatasetBase
from torch.optim.lr_scheduler import LRScheduler


import os
import math
import torch
import logging
import time
import numpy as np
from tqdm import tqdm
from xclib.utils.shortlist import Shortlist 
from xclib.utils.matrix import SMatrix
from torch.utils.data import DataLoader
from .batching import MySampler
from .model_base import ModelBase
from .evaluater import Evaluater


class ModelIS(ModelBase):
    """
    Generic class for models with implicit sampling
    
    Implicit sampling:
    - Negatives are not explicity sampled but selected
    from positive labels of other documents in the mini-batch
    - Also referred as in-batch or DPR in literature
    """
    def __init__(
            self,
            net: torch.nn.Module,
            criterion: _Loss,
            optimizer: Optimizer,
            schedular: LRScheduler,
            evaluater: Evaluater,
            model_dir: str,
            result_dir: str,
            shortlister: Shortlist,
            use_amp: bool = True,
            *args: Optional[Any],
            **kwargs: Optional[Any]
        ) -> None:
        """ 
        Args:
            net (torch.nn.Module): network object
            criterion (_Loss): object to compute loss
            optimizer (Optimizer): Optimizer
            schedular (LRScheduler): learning rate schedular
            evaluater (Evaluater): to evaluate
            model_dir (str): directory to save model etc.
            result_dir (str): directory to save results etc.
            shortlister (Shortlist): to be used at inference time
        """
        super().__init__(
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            schedular=schedular,
            evaluater=evaluater,
            model_dir=model_dir,
            result_dir=result_dir,
            use_amp=use_amp
        )
        self.shortlister = shortlister
        self.memory_bank = None

    def _create_data_loader(
        self,
        dataset: DatasetBase,
        prefetch_factor: int=5,
        batch_size: int=128,
        feature_t: str='sparse',
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
            feature_t (str, optional): Defaults to 'sparse'
                type of features.
            sampling_t (str, optional): Defaults to 'brute'.
                sampling type (used in creating data loader)
            num_workers (int, optional): Workers in dataloader. Defaults to 4.
            shuffle (bool, optional): Shuffle batches. Defaults to False.

        Returns:
            DataLoader: Data loader
        """
        batch_sampler = self.create_batch_sampler(
                dataset, batch_size, shuffle
            )
        dt_loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            num_workers=num_workers,
            collate_fn=self._create_collate_fn(
                feature_t, sampling_t, op_feature_t))
        return dt_loader

    def _compute_loss(self, y_hat: Tensor, batch: dict) -> Tensor:
        """Compute loss

        Args:
            pred (Tensor): predictions from the network
            batch_data (dict): dict containing (local) ground truth

        Returns:
            Tensor: computed loss
        """
        y = batch['Y'].to(y_hat.device)
        mask = batch['Y_mask']
        return self.criterion(
            y_hat,
            y,
            mask.to(y_hat.device) if mask is not None else mask)

    def update_order(self, data_loader: DataLoader) -> None:
        data_loader.batch_sampler.sampler.update_order(
            data_loader.dataset.indices_permutation())

    def _step_amp(self,
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
        with torch.amp.autocast(self.device):
            out, rep = self.net.forward(batch, precomputed_intermediate)
            loss = self._compute_loss(out, batch)
        if self.memory_bank is not None:
            ind = batch['indices']
            self.memory_bank[ind] = rep.detach().cpu().numpy()
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
        out, rep = self.net.forward(batch, precomputed_intermediate)
        if self.memory_bank is not None:
            ind = batch['indices']
            self.memory_bank[ind] = rep.detach().cpu().numpy()
        loss = self._compute_loss(out, batch)
        _loss = loss.item()
        loss.backward()
        self.optimizer.step()
        self.schedular.step()
        return _loss

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
            train_loader.dataset.step(self.memory_bank)
            tic = time.time()
            avg_loss = self._epoch(train_loader)
            toc = time.time()
            self.logger.info(
                f"Epoch: {epoch}, loss: {avg_loss:.6f}, time: {toc-tic:.2f} sec")
            if validation_loader is not None and epoch % validation_interval == 0:
                self.validate(validation_loader, epoch)
            self.update_order(train_loader)
        self.save_checkpoint(self.model_dir, epoch+1)

    def create_batch_sampler(self, dataset, batch_size, shuffle):
        if shuffle:
            order = dataset.indices_permutation()
        else:
            order = np.arange(len(dataset))
        return torch.utils.data.sampler.BatchSampler(
                MySampler(order), batch_size, False)

    def get_label_representations(self) -> Union[Tensor, ndarray]:
        raise NotImplementedError("")

    def _fit_shortlister(self, X):
        self.shortlister.fit(X)

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
        self._fit_shortlister(self.get_label_representations())
        top_k = min(k, data_loader.dataset.num_labels)
        predicted_labels = SMatrix(
            n_rows=data_loader.dataset.num_instances,
            n_cols=data_loader.dataset.num_labels,
            nnz=top_k)
        count = 0
        for batch_data in tqdm(data_loader):
            batch_size = batch_data['batch_size']
            emb = self.net.encode(batch_data['X'])
            # FIXME: the device may be different
            ind, vals = self.shortlister.query(emb, top_k)
            predicted_labels.update_block(
                count, ind.cpu().numpy(), vals.cpu().numpy())
            count += batch_size
            del batch_data
        return predicted_labels.data(), math.nan

    def save(self, model_dir: str, fname: str, *args: Any) -> None:
        """Save model on disk
        * uses suffix: _network.pkl for network

        Args:
            model_dir (str): save model into this directory
            fname (str): save model with this file name
        """
        super().save(model_dir, fname, args)
        fname = os.path.join(
            model_dir, fname+'.ann')
        self.shortlister.save(fname)

    def load(self, model_dir: str, fname: str, *args: Any) -> None:
        """Load model from disk
        * uses suffix: .ann for shortlister (ann index typically)

        Args:
            model_dir (str): load model from this directory
            fname (str): load model with this file name
        """
        super().load(model_dir, fname, args)
        fname = os.path.join(model_dir, fname+'.ann')
        self.shortlister.load(fname)


class XModelIS(ModelIS):
    """
    For models that do XC training with implicit sampling

    * XC training: classifiers and encoders (optionally) are trained    
    * Implicit sampling: negatives are not explicity sampled but
    selected from positive labels of other documents in the mini-batch

    """    
    def init_classifier(self, dataset, batch_size=128):
        #TODO
        raise NotImplementedError()

    def _init_memory_bank(self, dataset):
        # FIXME: fix the length of the vector
        self.memory_bank = np.zeros(
            (len(dataset), self.net.repr_dims),
            dtype='float32'
        )

    def fit(
        self,
        data_dir: str,
        dataset: str,
        trn_fname: str,
        val_fname: str,
        sampling_params: Namespace,
        trn_data: dict=None,
        val_data: dict=None,
        num_epochs: int=10,
        batch_size: int=128,
        num_workers: int=4,
        shuffle: bool=True,
        normalize_features=True,
        feature_t: str='dense',
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
        """
        TODO:
        * Support for frozen and pre-computed representation
        * Classifier initialization
        * Post-processing for inference
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
            feature_t=feature_t,
            sampling_t=sampling_params.type,
            sampling_params=sampling_params,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            surrogate_mapping=surrogate_mapping)
        train_loader = self._create_data_loader(
            train_dataset,
            batch_size=batch_size,
            feature_t=feature_t,
            sampling_t=sampling_params.type, # must be implicit
            num_workers=num_workers,
            shuffle=shuffle)
        self._init_memory_bank(train_dataset)
        validation_loader = None
        if validate_interval < num_epochs:
            self.logger.info("Loading validation data.")
            validation_dataset = self._create_dataset(
                os.path.join(data_dir, dataset),
                val_fname,
                data=val_data,
                mode='predict',
                feature_t=feature_t,
                normalize_features=normalize_features,
                normalize_labels=normalize_labels,
                surrogate_mapping=surrogate_mapping)
            validation_loader = self._create_data_loader(
                validation_dataset,
                feature_t=feature_t,
                batch_size=batch_size,
                num_workers=num_workers)
        self._fit(train_loader, validation_loader,
                  num_epochs, validate_interval)
        self.post_process_for_inference()

    def get_label_representations(self) -> Union[Tensor, ndarray]:
        return self.net.classifier.get_weights()

    def post_process_for_inference(self):
        self._fit_shortlister(self.get_label_representations())

    def _predict(self):
        raise NotImplementedError("")

    def predict(self): 
        raise NotImplementedError("")


class EModelIS(ModelIS):
    """
    For models that to train embedding or siamese models with implicit sampling

    * Siamese training: Encoders are trained    
    * Implicit sampling: negatives are not explicity sampled but
    selected from positive labels of other documents in the mini-batch

    """    
    def _init_memory_bank(self, dataset):
        self.memory_bank = np.zeros(
            (len(dataset), self.net.repr_dims),
            dtype='float32'
        )

    def fit(
        self,
        data_dir: str,
        dataset: str,
        trn_fname: str,
        val_fname: str,
        sampling_params: Namespace,
        trn_data: dict=None,
        val_data: dict=None,
        num_epochs: int=10,
        batch_size: int=128,
        num_workers: int=4,
        shuffle: bool=True,
        normalize_features=True,
        feature_t: str='dense',
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
        """
        TODO:
        * Post-processing for inference
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
            feature_t=feature_t,
            sampling_t=sampling_params.type,
            sampling_params=sampling_params,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            surrogate_mapping=surrogate_mapping)
        train_loader = self._create_data_loader(
            train_dataset,
            batch_size=batch_size,
            feature_t=feature_t,
            op_feature_t=feature_t,
            sampling_t=sampling_params.type, # must be implicit
            num_workers=num_workers,
            shuffle=shuffle)
        self._init_memory_bank(train_dataset)
        validation_loader = None
        if validate_interval < num_epochs:
            self.logger.info("Loading validation data.")
            validation_dataset = self._create_dataset(
                os.path.join(data_dir, dataset),
                val_fname,
                data=val_data,
                mode='predict',
                feature_t=feature_t,
                normalize_features=normalize_features,
                normalize_labels=normalize_labels,
                surrogate_mapping=surrogate_mapping)
            validation_loader = self._create_data_loader(
                validation_dataset,
                feature_t=feature_t,
                batch_size=batch_size,
                num_workers=num_workers)
        self._fit(train_loader, validation_loader,
                  num_epochs, validate_interval)
        self.post_process_for_inference(train_dataset)

    def get_label_representations(
            self, 
            dataset, 
            batch_size=128) -> Union[Tensor, ndarray]:
        self.net.eval()
        return self.get_embeddings(
            data=dataset.label_features.data,
            encoder=self.net.encode_lbl,
            batch_size=batch_size,
            feature_t=dataset.label_features._type
            )

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
        self._fit_shortlister(self.get_label_representations(data_loader.dataset))
        top_k = min(k, data_loader.dataset.num_labels)
        predicted_labels = SMatrix(
            n_rows=data_loader.dataset.num_instances,
            n_cols=data_loader.dataset.num_labels,
            nnz=top_k)
        count = 0
        for batch_data in tqdm(data_loader):
            batch_size = batch_data['batch_size']
            emb = self.net.encode(batch_data['X'])
            # FIXME: the device may be different
            ind, vals = self.shortlister.query(emb.cpu().numpy(), top_k)
            predicted_labels.update_block(count, ind, vals)
            count += batch_size
            del batch_data
        return predicted_labels.data(), math.nan

    def post_process_for_inference(self, dataset: DatasetBase) -> None:
        self._fit_shortlister(self.get_label_representations(dataset))

    def _predict(self):
        raise NotImplementedError("")

    def predict(self): 
        raise NotImplementedError("")
