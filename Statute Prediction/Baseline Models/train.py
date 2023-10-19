#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""Training and evaluation for BertMultiLabel"""

import argparse
import logging
import os
from collections import Counter
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from asl_loss import AsymmetricLoss
from torch.utils.data import DataLoader

import utils
from data_generator import BertMultiLabelDataset
from evaluate import evaluate
from metrics import metrics
from model.net import BertMultiLabel


def train_one_epoch(
    model, optimizer, loss_fn, data_loader, params, metrics, target_names, args
):
    # Set model to train
    model.train()
    m = nn.Sigmoid()

    criterion = loss_fn

    # For the loss of each batch
    loss_batch = []
    accumulate = utils.Accumulate()

    # Training Loop
    for i, (data, target, _) in enumerate(iter(data_loader)):
        logging.info(f"Training on batch {i + 1} with {len(data)} datapoints.")
        target = target.to(args.device)
        # Data is moved to relevant device in net.py after tokenization
        data = list(data)
        y_pred = model(data)
        loss = criterion(y_pred.float(), target.float())
        loss.backward()

        # Sub-batching behaviour to prevent memory overload
        if (i + 1) % params.update_grad_every == 0:
            optimizer.step()
            optimizer.zero_grad()
            loss_batch.append(loss.item())

        outputs_batch = (
            m(y_pred).data.cpu().detach().numpy() >= params.threshold
        ).astype(np.int32)

        targets_batch = (target.data.cpu().detach().numpy()).astype(np.int32)

        accumulate.update(outputs_batch, targets_batch)

        del data
        del target
        del outputs_batch
        del targets_batch
        del y_pred
        torch.cuda.empty_cache()

    else:
        # Last batch
        if (i + 1) % params.update_grad_every != 0:
            optimizer.step()
            optimizer.zero_grad()
            loss_batch.append(loss.item())

    outputs, targets = accumulate()

    summary_batch = {
        metric: metrics[metric](outputs, targets, target_names)
        for metric in metrics
    }
    summary_batch["loss_avg"] = sum(loss_batch) * 1.0 / len(loss_batch)

    return summary_batch


def train_and_evaluate(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    params,
    metrics,
    exp_dir,
    name,
    args,
    target_names,
    restore_file=None,
):
    # Default start epoch
    start_epoch = 0
    # Best train and val macro f1 variables
    best_train_macro_f1 = 0.0
    best_val_macro_f1 = 0.0

    # Load from checkpoint if any
    if restore_file is not None:
        restore_path = os.path.join(exp_dir, restore_file)

        logging.info(f"Found checkpoint at {restore_path}.")

        start_epoch = utils.load_checkpoint(restore_path, model, optimizer) + 1

        restore_file = None
        args.restore_file = None

    for epoch in range(start_epoch, params.num_epochs):
        logging.info(f"Logging for epoch {epoch}.")

        _ = train_one_epoch(
            model,
            optimizer,
            loss_fn,
            train_loader,
            params,
            metrics,
            target_names,
            args,
        )

        val_stats = evaluate(
            model, loss_fn, val_loader, params, metrics, args, target_names
        )

        train_stats = evaluate(
            model, loss_fn, train_loader, params, metrics, args, target_names
        )

        # Getting f1 val_stats

        train_macro_f1 = train_stats["f1"]["macro_f1"]
        is_train_best = train_macro_f1 >= best_train_macro_f1

        val_macro_f1 = val_stats["f1"]["macro_f1"]
        is_val_best = val_macro_f1 >= best_val_macro_f1

        logging.info(
            (
                f"val macro F1: {val_macro_f1:0.5f}\n"
                f"Train macro F1: {train_macro_f1:0.5f}\n"
                f"Avg val loss: {val_stats['loss_avg']:0.5f}\n"
                f"Avg train loss: {train_stats['loss_avg']:0.5f}\n"
            )
        )

        # Save val_stats
        train_json_path = os.path.join(
            exp_dir,
            "metrics",
            f"{name}",
            "train",
            f"epoch_{epoch + 1}_train_stats.json",
        )
        utils.save_dict_to_json(train_stats, train_json_path)

        val_json_path = os.path.join(
            exp_dir,
            "metrics",
            f"{name}",
            "val",
            f"epoch_{epoch + 1}_val_stats.json",
        )
        utils.save_dict_to_json(val_stats, val_json_path)

        # Saving best stats
        if is_train_best:
            best_train_macro_f1 = train_macro_f1
            train_stats["epoch"] = epoch + 1

            best_json_path = os.path.join(
                exp_dir, "metrics", f"{name}", "train", "best_train_stats.json"
            )
            utils.save_dict_to_json(train_stats, best_json_path)

        if is_val_best:
            best_val_macro_f1 = val_macro_f1
            val_stats["epoch"] = epoch + 1

            best_json_path = os.path.join(
                exp_dir, "metrics", f"{name}", "val", "best_val_stats.json"
            )
            utils.save_dict_to_json(val_stats, best_json_path)

            logging.info(
                (
                    f"New best macro F1: {best_val_macro_f1:0.5f} "
                    f"Train macro F1: {train_macro_f1:0.5f} "
                    f"Avg val loss: {val_stats['loss_avg']} "
                    f"Avg train loss: {train_stats['loss_avg']}."
                )
            )

        state = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optim_dict": optimizer.state_dict(),
        }

        utils.save_checkpoint(
            state,
            is_val_best,
            os.path.join(exp_dir, "model_states", f"{name}"),
            (epoch + 1) % params.save_every == 0,
        )

    # For the last epoch

    utils.save_checkpoint(
        state,
        is_val_best,
        os.path.join(exp_dir, "model_states", f"{name}"),
        True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_dirs",
        nargs="+",
        type=str,
        default=["data/"],
        help=("Directory containing training and validation cases."),
    )
    parser.add_argument(
        "-t",
        "--targets_paths",
        nargs="+",
        type=str,
        default=["targets/targets.json"],
        help="Path to target files.",
    )
    parser.add_argument(
        "-x",
        "--exp_dir",
        default="experiments/",
        help=(
            "Directory to load parameters "
            " from and save metrics and model states"
        ),
    )
    parser.add_argument(
        "-n", "--name", type=str, required=True, help="Name of model"
    )
    parser.add_argument(
        "-p",
        "--params",
        default="params.json",
        help="Name of params file to load from exp+_dir",
    )
    parser.add_argument(
        "-de", "--device", type=str, default="cuda", help="Device to train on."
    )
    parser.add_argument(
        "-id",
        "--device_id",
        type=int,
        default=0,
        help="Device ID to run on if using GPU.",
    )
    parser.add_argument(
        "-r", "--restore_file", default=None, help="Restore point to use."
    )
    parser.add_argument(
        "-ul",
        "--unique_labels",
        type=str,
        default=None,
        help="Labels to use as targets.",
    )
    parser.add_argument(
        "-lm",
        "--model_name",
        type=str,
        default="allenai/longformer-base-4096",
        help="BERT variant to use as model.",
    )

    args = parser.parse_args()

    # Setting logger
    utils.set_logger(os.path.join(args.exp_dir, f"{args.name}"))

    # Selecting correct device to train and evaluate on
    if not torch.cuda.is_available() and args.device == "cuda":
        logging.info("No CUDA cores/support found. Switching to cpu.")
        args.device = "cpu"

    if args.device == "cuda":
        args.device = f"cuda:{args.device_id}"

    logging.info(f"Device is {args.device}.")

    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    # Loading parameters
    params_path = os.path.join(args.exp_dir, "params", f"{args.params}")
    assert os.path.isfile(params_path), f"No params file at {params_path}"
    params = utils.Params(params_path)

    # Setting seed for reproducability
    torch.manual_seed(47)
    if "cuda" in args.device:
        torch.cuda.manual_seed(47)

    # Setting data paths
    train_paths = []
    val_paths = []
    for path in args.data_dirs:
        train_paths.append(os.path.join(path, "train"))
        val_paths.append(os.path.join(path, "validation"))

    # Datasets
    train_dataset = BertMultiLabelDataset(
        data_paths=train_paths,
        targets_paths=args.targets_paths,
        unique_labels=args.unique_labels,
    )

    val_dataset = BertMultiLabelDataset(
        data_paths=val_paths,
        targets_paths=args.targets_paths,
        unique_labels=args.unique_labels,
    )

    logging.info(f"Training with {len(train_dataset.unique_labels)} targets")
    logging.info(f"Training on {len(train_dataset)} datapoints")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=params.batch_size, shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=params.batch_size,
        shuffle=True,
    )

    model = BertMultiLabel(
        labels=train_dataset.unique_labels,
        device=args.device,
        hidden_size=params.hidden_dim,
        max_length=params.max_length,
        model_name=args.model_name,
        truncation_side=params.truncation_side,
    )

    model.to(args.device)

    # Defining optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=params.lr)

    logging.info("Calculating positive weights for loss")

    target_counts = Counter(
        chain.from_iterable(
            train_dataset.targets_dict[v] for v in train_dataset.idx.values()
        )
    )
    logging.info(f"Number of positives for classes: {target_counts}")

    pos_weight = [
        (1.0 - target_counts[k] * 1 / len(train_dataset))
        * (len(train_dataset) * 1.0 / target_counts.get(k, 1))
        for k in train_dataset.unique_labels
    ]

    pos_weight = torch.FloatTensor(pos_weight)
    pos_weight.to(args.device)
    logging.info(f"Calculated positive weights are: {pos_weight}")
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weight).to(
        args.device
    )

    train_and_evaluate(
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        params,
        metrics,
        args.exp_dir,
        args.name,
        args,
        train_dataset.unique_labels,
        restore_file=args.restore_file,
    )

    logging.info("=" * 80)


if __name__ == "__main__":
    main()
