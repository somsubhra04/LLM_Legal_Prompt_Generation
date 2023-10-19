#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""Evaluation script for BertMultiLabel"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from data_generator import BertMultiLabelDataset
from metrics import metrics
from model.net import BertMultiLabel


def evaluate(model, loss_fn, data_loader, params, metrics, args, target_names):
    if args.restore_file is not None:
        # Loading trained model
        logging.info(f"Found checkpoint at {args.restore_file}. Loading.")
        _ = utils.load_checkpoint(args.restore_file, model, device_id=0) + 1
        args.restore_file = None

    # Set model to eval mode
    model.eval()
    m = nn.Sigmoid()

    # Accumulate data of batches
    accumulate = utils.Accumulate()
    loss_batch = []
    preds = {}

    criterion = loss_fn

    for data, target, idx in iter(data_loader):
        target = target.to(args.device)
        data = list(data)
        y_pred = model(data)

        loss = criterion(y_pred.float(), target.float())

        outputs_batch = (
            m(y_pred).data.cpu().numpy() > params.threshold
        ).astype(np.int32)
        targets_batch = (target.data.cpu().numpy()).astype(np.int32)

        accumulate.update(outputs_batch, targets_batch)
        loss_batch.append(loss.item())

        pred_idx = [
            [i for i, val in enumerate(pred) if val != 0.0]
            for pred in outputs_batch
        ]
        pred_names = [[target_names[j] for j in pred] for pred in pred_idx]
        for i, pred in zip(idx, pred_names):
            preds[i] = pred

        del data
        del target
        del y_pred
        del outputs_batch
        del targets_batch
        del pred

    output, targets = accumulate()

    summary_batch = {
        metric: metrics[metric](output, targets, target_names)
        for metric in metrics
    }
    summary_batch["preds"] = preds

    summary_batch["loss_avg"] = sum(loss_batch) * 1.0 / len(loss_batch)

    return summary_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_paths",
        nargs="+",
        type=str,
        default=["data/"],
        help="Directory containing test cases.",
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
        help="Longformer variant to use as model.",
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

    logging.info("Final arguments are:")
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

    # Datasets
    test_dataset = BertMultiLabelDataset(
        data_paths=args.data_paths,
        targets_paths=args.targets_paths,
        unique_labels=args.unique_labels,
    )

    logging.info(f"[DATASET] Using {len(test_dataset.unique_labels)} targets")
    logging.info(f"[DATASET] Test set contains {len(test_dataset)} datapoints")

    # Dataloader
    test_loader = DataLoader(
        test_dataset, batch_size=params.batch_size, shuffle=True
    )

    model = BertMultiLabel(
        labels=test_dataset.unique_labels,
        device=args.device,
        hidden_size=params.hidden_dim,
        max_length=params.max_length,
        model_name=args.model_name,
        truncation_side=params.truncation_side,
    )

    model.to(args.device)

    # Defining optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

    test_stats = evaluate(
        model,
        loss_fn,
        test_loader,
        params,
        metrics,
        args,
        test_dataset.unique_labels,
    )

    json_path = os.path.join(
        args.exp_dir, "metrics", f"{args.name}", "test", "test_stats.json"
    )
    utils.save_dict_to_json(test_stats, json_path)

    logging.info("=" * 80)


if __name__ == "__main__":
    main()
