#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""Metrics to be calculated for the model."""

from typing import MutableSequence

import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs


def custom_f1(outputs_batch: MutableSequence,
              targets_batch: MutableSequence, target_names: list) -> dict:
    """Calculate per class and macro F1 between the given predictions
    and targets

    Parameters
    ----------
    outputs_batch : MutableSequence
        Predictions of a batch.
    targets_batch : MutableSequence
        Targets of the batch.
    target_names  : list[str]
        Names of targets.

    Returns
    -------
    scores : dict
        Dictionary containing the metric values.

    """

    per_class_prec = []
    per_class_rec = []

    num_classes = targets_batch.shape[-1]

    for cls in range(num_classes):
        tp = np.dot(targets_batch[:, cls], outputs_batch[:, cls])
        pp = np.sum(outputs_batch[:, cls])
        p = np.sum(targets_batch[:, cls])
        prec = tp/pp if pp != 0 else 0
        rec = tp/p if p != 0 else 0

        per_class_prec.append(prec)
        per_class_rec.append(rec)

    den = [per_class_prec[i] + per_class_rec[i]
           for i in range(len(per_class_rec))]
    num = [2 * (per_class_prec[i] * per_class_rec[i])
           for i in range(len(per_class_rec))]

    per_class_f1 = [num_val * 1./den_val if den_val != 0 else 0
                    for num_val, den_val in zip(num, den)]

    macro_f1 = sum(per_class_f1) * 1./len(per_class_f1)

    # Converting metrics to dictionaries for easier understanding
    per_class_prec = {
            k: per_class_prec[i] for i, k in enumerate(target_names)}
    per_class_rec = {
            k: per_class_rec[i] for i, k in enumerate(target_names)}
    per_class_f1 = {
            k: per_class_f1[i] for i, k in enumerate(target_names)}

    scores = {
        'precision': per_class_prec,
        'recall': per_class_rec,
        'f1': per_class_f1,
        'macro_f1': macro_f1,
        }

    return scores


def f1(outputs_batch: MutableSequence,
       targets_batch: MutableSequence, target_names: list) -> dict:
    """Calculate per class and macro F1 between the given predictions
    and targets

    Parameters
    ----------
    outputs_batch : MutableSequence
        Predictions of a batch.
    targets_batch : MutableSequence
        Targets of the batch.
    target_names  : list[str]
        Names of targets.

    Returns
    -------
    scores : dict
        Dictionary containing the metric values.

    """
    class_metrics = prfs(targets_batch, outputs_batch, average=None)
    per_class_prec, per_class_rec, per_class_f1, per_class_sup = class_metrics

    macro_metrics = prfs(targets_batch, outputs_batch, average='macro')
    macro_prec, macro_rec, macro_f1, macro_sup = macro_metrics

    micro_metrics = prfs(targets_batch, outputs_batch, average='micro')
    micro_prec, micro_rec, micro_f1, micro_sup = micro_metrics

    # Converting metrics to dictionaries for easier understanding
    per_class_prec = {
            k: float(per_class_prec[i]) for i, k in enumerate(target_names)}
    per_class_rec = {
            k: float(per_class_rec[i]) for i, k in enumerate(target_names)}
    per_class_f1 = {
            k: float(per_class_f1[i]) for i, k in enumerate(target_names)}
    per_class_sup = {
            k: float(per_class_sup[i]) for i, k in enumerate(target_names)}

    scores = {
        'precision': per_class_prec,
        'recall': per_class_rec,
        'f1': per_class_f1,
        'sup': per_class_sup,
        'macro_prec': float(macro_prec) if macro_prec is not None else macro_prec,
        'macro_rec': float(macro_rec) if macro_rec is not None else macro_rec,
        'macro_f1': float(macro_f1) if macro_f1 is not None else macro_f1,
        'macro_sup': float(macro_sup) if macro_sup is not None else macro_sup,
        'micro_prec': float(micro_prec) if micro_prec is not None else micro_prec,
        'micro_rec': float(micro_rec) if micro_rec is not None else micro_rec,
        'micro_f1': float(micro_f1) if micro_f1 is not None else micro_f1,
        'micro_sup': float(micro_sup) if micro_sup is not None else micro_sup,
        }

    return scores


metrics = {
        'f1': f1,
        }
