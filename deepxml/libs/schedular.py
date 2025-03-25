from argparse import Namespace

import torch


def construct_schedular(
        optimizer: torch.optim.Optimizer,
        args: Namespace) -> torch.optim.lr_scheduler:
    """Construct schedular with warmup and linear decay

    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        args (Namespace): object with following attributes 
            warmup_steps, num_epochs, num_points, batch_size

    Returns:
        torch.optim.lr_scheduler: Schedular object
    """
    return get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_epochs*(
            args.num_points/args.batch_size))


def get_linear_schedule_with_warmup(
        optimizer: torch.optim.Optimizer, 
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int=-1) -> torch.optim.lr_scheduler:
    """ Create a schedule with a learning rate that decreases linearly from
    the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.


    Args:
        optimizer (torch.optim.Optimizer): 
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        last_epoch (int, optional): Defaults to -1.
            The index of the last epoch when resuming training.

    Returns:
        torch.optim.lr_scheduler: Schedular object
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        num = num_training_steps - current_step
        den = max(1, num_training_steps - num_warmup_steps)
        return max(0.0, float(num) / float(den))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
