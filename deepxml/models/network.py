from typing import Union
from torch import Tensor
from torch.nn import Module

import os
import math
import torch
import torch.nn as nn
from argparse import Namespace
import torch.nn.functional as F
from .modules import parse_json, construct_module
from .utils import cosine_sim, ip_sim


def _to_device(
        x: Union[Tensor, list, dict, None],
        device: str) -> Union[Tensor, list, dict, None]:
    """Transfer a Tensor to device with support for list, tuples, dict or dict

    Args:
        x (Union[Tensor, list, dict, None]): input which needs to be transfeered
        device (str): target device

    Returns:
        Union[Tensor, list, dict, None]: item transferred to device
    """
    if x is None:
        return None
    elif isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    elif isinstance(x, (tuple, list)):
        return [_to_device(it, device) for it in x]
    else:
        return x.to(device)


class BaseNetwork(Module):
    """BaseNetwork: Base class for different networks

    Components:
    * encoder: an encoder layer to encode the inputs (e.g. queries)
    * encoder_lbl: an encoder layer to encode the outputs (e.g. labels)
        - encoder_lbl will be same as encoder unless it is passed explicitly
    * classifier: explicit classifiers (e.g., 1-vs-All classifiers)
        - Identity op as classifier if is not pass
    """

    def __init__(
            self, 
            config: str, 
            args: Namespace=Namespace(),
            encoder: Module=None, 
            encoder_lbl: Module=None,
            classifier: Module=None, 
            device="cuda") -> None:
        """
        Args:
            config (str): json file containing the network components
            args (Namespace): Values of placeholders can be taken from args
                * "#ARGS.x;" value in config will be replaced with args.x
            encoder (Module or None): encoder module for inputs or queries
                - to encode given (raw) representation
            encoder_lbl (Module or None): encoder module for output or labels
                - to encode given (raw) representation
            classifier (Module or None): Classifier module 
                - can be identity or None if required
            device (str, optional): device for network. Defaults to "cuda".
        """
        super(BaseNetwork, self).__init__()
        self.device = torch.device(device)
        if config is not None:
            self._construct_from_config(parse_json(config, args))
        elif encoder is not None:
            self._construct_from_module(encoder, encoder_lbl, classifier)
        else:
            raise NotImplementedError(
                "Either of modules or config must be valid")

    def _construct_from_module(
            self, 
            encoder: Module,
            encoder_lbl: Module, 
            classifier: Module, 
        ) -> Module:
        self.encoder = encoder
        self.classifier = classifier
        if self.encoder_lbl is not None:
            self.encoder_lbl = encoder_lbl
        else:
            self.encoder_lbl = encoder
        if hasattr(self.encoder, 'repr_dims'):
            self._repr_dims = self.encoder.repr_dims

    def _construct_from_config(
            self, 
            config: dict, 
            device: str="cuda") -> Module:
        """Construct the class from config

        Args:
            config (str): a json file consisting the archietcture
            device (str, optional): device for model. Defaults to "cuda".

        Returns:
            Module: class instance
        """
        self.encoder = self._construct_encoder(config['encoder'])
        #TODO: See if it is better for them to be None or as identity
        if 'classifier' in config:
            self.classifier = self._construct_classifier(config['classifier'])

        if 'encoder_lbl' in config:
           self.encoder_lbl = self._construct_encoder(config['encoder_lbl'])
        else:
            self.encoder_lbl = self.encoder

        if hasattr(self.encoder, 'repr_dims'):
            self._repr_dims = self.encoder.repr_dims

        if 'repr_dims' in config:
            self._repr_dims = int(config['repr_dims'])

    def _construct_encoder(self, config: dict) -> Module:
        # Construct encoder from dictionary
        # (useful in case you want to do some custom stuff)
        return self._construct_module(config)

    def _construct_classifier(self, config: dict) -> Module:
        # Construct classifier from dictionary
        # (useful in case you want to do some custom stuff)
        return self._construct_module(config)

    @classmethod
    def from_config(
        cls,
        config: str,
        args:Namespace=Namespace(),
        device: str="cuda") -> Module:
        """Construct the class from config

        Args:
            config (str): a json file consisting the archietcture
            device (str, optional): device for model. Defaults to "cuda".

        Returns:
            Module: class instance
        """
        return cls(config=config, args=args, device=device)

    @classmethod
    def from_modules(
        cls, 
        encoder: Module, 
        encoder_lbl: Module,
        classifier: Module, 
        device: str="cuda") -> Module:
        """Construct the class from already constructed modules
        * useful when encoder and classifiers are already constructed

        Args:
            encoder (Module): To encode given (raw) representation
            classifier (Module): Classifier module (can be identity if required)
            device (str, optional): device for model. Defaults to "cuda".

        Returns:
            Module: class instance
        """
        return cls(None, encoder, encoder_lbl, classifier, device=device)

    def _construct_module(self, config: str=dict) -> Module:
        if config is None:
            return nn.Identity()
        return construct_module(config)

    @property
    def repr_dims(self) -> int:
        """Representation dimension
        """
        return self._repr_dims

    @repr_dims.setter
    def repr_dims(self, dims: int) -> None:
        self._repr_dims = dims

    def encode(self, x: tuple, *args, **kwargs) -> Tensor:
        """Encode an item using the given network

        Args:
            x (tuple): #TODO

        Returns:
            torch.Tensor: Encoded item
        """
        return self.encoder(_to_device(x, self.device))

    def forward(self, batch, *args):
        """Forward pass

        * Assumes features are dense if X_w is None
        * By default classifier is identity op

        Args:
            batch (dict): A dictionary containing features or 
                tokenized representation

        Returns:
            torch.Tensor: output of the network (typically logits)
        """
        return self.classifier(self.encode(batch['X']))

    def initialize(self) -> None:
        """Initialize embeddings from existing ones"""
        raise NotImplementedError()
        
    def purge(self, fname: str) -> None:
        """Purge the saved model

        Args:
            fname (str): model file path on the disk
        """
        if os.path.isfile(fname):
            os.remove(fname)

    @property
    def num_params(self, ignore_fixed: bool=False) -> int:
        """Number of parameters in the network

        Args:
            ignore_fixed (bool, optional): Defaults to False.
                Ignore the parameters where gradients are false

        Returns:
            int: Number of parameters in the network
        """
        if ignore_fixed:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    @property
    def model_size(self) -> float:  # Assumptions: 32bit floats
        """Get model size as per 32 bit floats
        """
        return self.num_params * 4 / math.pow(2, 20)

    def __repr__(self):
        return f"(Encoder): {self.encoder}\n(Label Encoder): {self.encoder_lbl}\n(Classifier): {self.classifier}"


class XMLNetwork(BaseNetwork):
    """
    Class to train extreme classifiers in brute force manner
    """
    pass


class XMLNetworkIS(BaseNetwork):
    """
    Class to train extreme classifiers with shared shortlist
    """
    def forward(self, batch, *args):
        """Forward pass

        * Assumes features are dense if X_w is None

        Args:
            batch (dict): A dictionary containing features or 
                tokenized representation and shared label shortlist

        Returns:
            torch.Tensor: output of the network (typically logits)
        """
        X = self.encode(_to_device(batch['X'], self.device))
        return self.classifier(
            X, _to_device(batch['Y_s'], self.device)), X


class SiameseNetworkIS(BaseNetwork):
    """
    Class to train encoder with shared shortlist
    """
    def __init__(
            self, 
            config: str, 
            args: Namespace=Namespace(),
            encoder: Module=None, 
            encoder_lbl: Module=None,
            classifier: Module=None, 
            device="cuda") -> None:
        """
        Args:
            config (str): json file containing the network components
            args (Namespace): Values of placeholders can be taken from args
                * "#ARGS.x;" value in config will be replaced with args.x
            encoder (Module or None): encoder module for inputs or queries
                - to encode given (raw) representation
            encoder_lbl (Module or None): encoder module for output or labels
                - to encode given (raw) representation
            classifier (Module or None): Classifier module 
                - can be identity or None if required
            device (str, optional): device for network. Defaults to "cuda".
        """
        super(SiameseNetworkIS, self).__init__(
            config=config,
            args=args,
            encoder=encoder,
            encoder_lbl=encoder_lbl,
            classifier=classifier,
            device=device)
        if not hasattr(args, 'metric'):
            args.metric = 'cosine'
        self.metric = args.metric
        self.similarity = self._setup_metric(args.metric)

    def _setup_metric(self, metric):
        if metric == 'cosine':
            return cosine_sim       
        elif metric == 'ip':
            return ip_sim 
        else:
            raise NotImplementedError("Unknown metric!")      

    def encode_lbl(self, x: tuple) -> Tensor:
        """Encode an item using the given network

        Args:
            x (tuple): #TODO

        Returns:
            torch.Tensor: Encoded item
        """
        return self.encoder_lbl(_to_device(x, self.device))

    def forward(self, batch, *args):
        """Forward pass

        * Assumes features are dense if X_w is None

        Args:
            batch (dict): A dictionary containing features or 
                tokenized representation and shared label shortlist

        Returns:
            torch.Tensor: output of the network (typically logits)
        """
        X = self.encode(_to_device(batch['X'], self.device))
        Z = self.encode_lbl(_to_device(batch['Z'], self.device))
        return self.similarity(X, Z), X
