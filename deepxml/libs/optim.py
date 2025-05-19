from argparse import Namespace

import torch
import transformers

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter(action='ignore', category=FutureWarning)


def _get_no_decay(nd):
    if nd is None:
        no_decay = ['bias', 'LayerNorm.weight']
    else:
        try:
            no_decay = eval(nd)
        except NameError:
            no_decay = [nd]        
    return no_decay


def _get_optimizer_class(optim):
    if optim == 'AdamW': # there is some difference in AdamW of torch
        optimizer_cls = transformers.AdamW
    else:
        try:
            optimizer_cls = getattr(torch.optim, optim)
        except AttributeError:
            raise NotImplementedError(
                f"Optimizer '{optim}' is not available in torch.optim")
    return optimizer_cls


class MixedOptimizer(torch.optim.Optimizer):
    """Optimizer that handles sparse gradients for some parameters and dense for others."""

    def __init__(self, dense_params, sparse_params, dense_optimizer_cls, sparse_optimizer_cls, dense_kwargs, sparse_kwargs):
        """
        Args:
            dense_params (iterable): Parameters with dense gradients.
            sparse_params (iterable): Parameters with sparse gradients.
            dense_optimizer_cls (type): Optimizer class for dense gradients (e.g., torch.optim.Adam).
            sparse_optimizer_cls (type): Optimizer class for sparse gradients (e.g., torch.optim.SparseAdam).
            dense_kwargs (dict): Arguments for the dense optimizer.
            sparse_kwargs (dict): Arguments for the sparse optimizer.
        """
        self.dense_optimizer = dense_optimizer_cls(dense_params, **dense_kwargs)
        self.sparse_optimizer = sparse_optimizer_cls(sparse_params, **sparse_kwargs)
        self.optimizers = [self.dense_optimizer, self.sparse_optimizer]

    def step(self, closure=None):
        """Performs a single optimization step for both dense and sparse optimizers."""
        loss = None
        if closure is not None:
            loss = closure()
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    def zero_grad(self, set_to_none=False):
        """Clears the gradients of all optimized parameters."""
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Returns the state of the optimizer as a dict."""
        return {
            'dense_optimizer': self.dense_optimizer.state_dict(),
            'sparse_optimizer': self.sparse_optimizer.state_dict()
        }

    @property
    def param_groups(self):
        """Combines parameter groups from both dense and sparse optimizers."""
        return self.dense_optimizer.param_groups + self.sparse_optimizer.param_groups

    def load_state_dict(self, state_dict):
        """Loads the optimizer state."""
        self.dense_optimizer.load_state_dict(state_dict['dense_optimizer'])
        self.sparse_optimizer.load_state_dict(state_dict['sparse_optimizer'])


def construct_mixed_optimizer(net, args):
    no_decay = _get_no_decay(getattr(args, 'no_decay', None))
    weight_decay = getattr(args, 'weight_decay', 0.0)
    learning_rate = args.learning_rate

    sparse_parameters = [
        {'params': [p for n, p in net.named_parameters() if 'classifier' in n]}
    ]

    dense_parameters = [
        {'params': [p for n, p in net.named_parameters() if 'classifier' not in n and not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in net.named_parameters() if 'classifier' not in n and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    return MixedOptimizer(
        dense_params=dense_parameters,
        sparse_params=sparse_parameters,
        dense_optimizer_cls=_get_optimizer_class('AdamW'),
        sparse_optimizer_cls=_get_optimizer_class('SparseAdam'),
        dense_kwargs={'weight_decay': weight_decay, 'lr': learning_rate},
        sparse_kwargs={'lr': learning_rate}
    )


def construct_optimizer(
        net: torch.nn.Module, 
        args: Namespace) -> torch.optim.Optimizer:
    """Construct Optimizer dynamically based on string name

    Args:
        net (torch.nn.Module): network that will be trained
        args (Namespace): object with learning_rate, weight_decay attributes

    Raises:
        NotImplementedError: For unknown optimizer

    Returns:
        torch.optim: Optimizer
    """
    weight_decay = getattr(args, 'weight_decay', 0.0)
    learning_rate = args.learning_rate
    
    optim_cls = _get_optimizer_class(args.optim)
    no_decay = _get_no_decay(getattr(args, 'no_decay', None))

    if no_decay:
        parameters = [
            {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    else:
        parameters = net.parameters(
            lr=learning_rate,
            weight_decay=weight_decay
        )

    return optim_cls(parameters, **{'lr': learning_rate, 'eps': 1e-06})
