from typing import Union
from torch import Tensor
from torch.nn import Module

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import cModule


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
    """DeepXMLBase: Base class for different networks
    * Identity op as classifier by default
    (derived class should implement it's own classifier)
    * embedding and classifier shall automatically transfer
    the vector to the appropriate device
    """

    def __init__(
            self, 
            config: str, 
            encoder: Module=None, 
            classifier: Module=None, 
            device="cuda") -> None:
        """
        Args:
            config (str): json file containing the network components
            encoder (Module or None): encoder module
                - to encode given (raw) representation
            classifier (Module or None): Classifier module 
                - can be identity or None if required
            device (str, optional): device for network. Defaults to "cuda".
        """
        super(BaseNetwork, self).__init__()
        self.device = torch.device(device)
        if config is not None:
            self._construct_from_config(config)
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

    def _construct_from_config(self, config: str, device: str="cuda") -> Module:
        """Construct the class from config

        Args:
            config (str): a json file consisting the archietcture
            device (str, optional): device for model. Defaults to "cuda".

        Returns:
            Module: class instance
        """
        self.encoder = self._construct_module(config)
        self.classifier = self._construct_classifier(config)

    @classmethod
    def from_config(cls, config: str, device: str="cuda") -> Module:
        """Construct the class from config

        Args:
            config (str): a json file consisting the archietcture
            device (str, optional): device for model. Defaults to "cuda".

        Returns:
            Module: class instance
        """
        return cls(config=config, device=device)

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

    def _construct_classifier(self, *args, **kwargs) -> Module:
        return nn.Identity()

    def _construct_module(self, config: str=None) -> Module:
        if config is None:
            return nn.Identity()
        return cModule(config)

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