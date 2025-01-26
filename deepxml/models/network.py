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
    * Identity op as classifier by default
    (derived class should implement it's own classifier)
    * embedding and classifier shall automatically transfer
    the vector to the appropriate device
    """

    def __init__(
            self, 
            config: str, 
            args: Namespace=Namespace(),
            encoder: Module=None, 
            classifier: Module=None, 
            device="cuda") -> None:
        """
        Args:
            config (str): json file containing the network components
            args (Namespace): Values of placeholders can be taken from args
                * "#ARGS.x;" value in config will be replaced with args.x
            encoder (Module or None): encoder module
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
            self._construct_from_module(encoder, classifier)
        else:
            raise NotImplementedError(
                "Either of modules or config must be valid")

    def _construct_from_module(
            self, 
            encoder: Module, 
            classifier: Module, 
        ) -> Module:
        self.encoder = encoder
        self.classifier = classifier

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
        self.classifier = self._construct_classifier(config['classifier'])

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
        return cls(None, encoder, classifier, device=device)

    def _construct_module(self, config: str=dict) -> Module:
        if config is None:
            return nn.Identity()
        return construct_module(config)

    @property
    def representation_dims(self) -> int:
        """Representation dimension
        """
        return self._repr_dims

    @representation_dims.setter
    def representation_dims(self, dims: int) -> None:
        self._repr_dims = dims

    def encode(self, x: tuple) -> Tensor:
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
        return f"{self.encoder}\n(Classifier): {self.classifier}"


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
        X = self.encode(batch['X'])
        return self.classifier(X, batch['Y_s']), X
