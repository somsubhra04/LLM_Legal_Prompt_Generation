#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""Utilities for BertMultiLabel"""

import errno
import json
import logging
import os
from time import strftime

import numpy as np
import torch
import torch.nn as nn


class Params:
    """Class that loads hyperparameters from a json file."""

    def __init__(self, json_path):
        with open(json_path, "r") as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @property
    def dict(self):
        """Provide dictionary-like access to hyperparameters."""
        return self.__dict__


class Accumulate:
    """Maintain all data used in an epoch for metrics calculation."""

    def __init__(self):
        self.output_batch = []
        self.targets_batch = []

    def update(self, output_batch, targets_batch):
        self.output_batch.extend(output_batch.tolist())
        self.targets_batch.extend(targets_batch.tolist())

    def __call__(self):
        return (
            np.stack(self.output_batch, axis=0),
            np.stack(self.targets_batch, axis=0),
        )


def load_checkpoint(restore_path, model, optimizer=None, device_id=None):
    if not (os.path.exists(restore_path)):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), restore_path
        )

    if device_id is None:
        ckpt = torch.load(restore_path)
    else:
        ckpt = torch.load(restore_path, map_location=f"cuda:{device_id}")

    model.load_state_dict(ckpt["state_dict"])

    if optimizer:
        optimizer.load_state_dict(ckpt["optim_dict"])

    return ckpt["epoch"]


def save_dict_to_json(dict_obj, save_path):
    if not (os.path.exists(os.path.split(save_path)[0])):
        os.makedirs(os.path.split(save_path)[0])

    with open(save_path, "w") as f:
        json.dump(dict_obj, f, indent=4)


def save_checkpoint(state, is_best, save_path, to_save=False):
    if not (os.path.exists(save_path)):
        os.makedirs(save_path)

    if is_best:
        torch.save(state, os.path.join(save_path, "best.pth.tar"))

    if to_save:
        torch.save(
            state, os.path.join(save_path, f"epoch_{state['epoch']}.pth.tar")
        )


def set_logger(log_path: str):
    timestamp = strftime("%Y-%m-%d-%H-%M-%S")
    log_path = log_path + "_" + timestamp + ".log"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s: [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
            )
        )
        logger.addHandler(file_handler)

        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s : [%(levelname)s] %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(stream_handler)
