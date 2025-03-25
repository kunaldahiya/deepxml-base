from argparse import Namespace

import torch
import transformers


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
    args.weight_decay = 0.0 if not hasattr(args, 'weight_decay') else args.weight_decay

    if args.optim == 'AdamW': # there is some difference in AdamW of torch
        optimizer_cls = transformers.AdamW
    else:
        try:
            optimizer_cls = getattr(torch.optim, args.optim)
        except AttributeError:
            raise NotImplementedError(
                f"Optimizer '{args.optim}' is not available in torch.optim")

    if hasattr(args, 'no_decay'):
        try:
            no_decay = eval(args.no_decay)
        except NameError:
            no_decay = [args.no_decay]
    else:
        no_decay = ['bias', 'LayerNorm.weight']

    if no_decay:
        parameters = [
            {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    else:
        parameters = net.parameters(
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

    return optimizer_cls(parameters, **{'lr': args.learning_rate, 'eps': 1e-06})
