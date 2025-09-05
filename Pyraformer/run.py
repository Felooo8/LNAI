"""Utility entry-point for programmatic Pyraformer training.

This wrapper exposes a :func:`run` function that accepts an ``argparse``
``Namespace`` (the same options as ``single_step_main``) and executes the
standard training loop, returning a dictionary with the best validation
metrics.
"""

import os
import torch
import torch.optim as optim
import argparse

import pyraformer.Pyraformer_SS as Pyraformer

from .single_step_main import (
    arg_parser,
    get_dataset_parameters,
    train,
)


def run(args: argparse.Namespace):
    """Train Pyraformer using the given configuration ``args``.

    Parameters
    ----------
    args: ``argparse.Namespace``
        Configuration matching ``single_step_main.arg_parser``.

    Returns
    -------
    dict
        Mapping of metric name to value from the best epoch.
    """

    args = get_dataset_parameters(args)
    if isinstance(args.window_size, str):
        args.window_size = eval(args.window_size)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Pyraformer.Model(args)
    model.to(args.device)

    model_save_dir = os.path.join('models/SingleStep', args.dataset)
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, 'best_model.pth')

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)

    index_names, best_metrics = train(model, optimizer, scheduler, args, model_path)
    return {name: float(metric) for name, metric in zip(index_names, best_metrics)}


def main():
    args = arg_parser()
    run(args)


if __name__ == "__main__":
    main()

