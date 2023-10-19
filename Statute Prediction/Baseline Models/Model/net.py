#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""BERT-based model for multi-label classification."""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class BertMultiLabel(nn.Module):

    """Auto-based model for multi-label classification"""

    def __init__(
        self,
        labels,
        device,
        hidden_size=768,
        max_length=4096,
        model_name="allenai/longformer-base-4096",
        truncation_side="right",
        mode="train",
    ):
        super(BertMultiLabel, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.max_length = max_length
        self.labels = [re.sub(r"[^A-Za-z0-9_]", "", label) for label in labels]
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name)
        self.truncation_side = truncation_side
        self.mode = mode
        # Keeping the tokenizer here makes the model better behaved
        # as opposed to using it in the DataLoader
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, truncation_side=self.truncation_side
        )

        self.prediction = nn.ModuleDict(
            {
                k: nn.Linear(
                    in_features=self.hidden_size,
                    out_features=1,
                    bias=True,
                )
                for k in self.labels
            }
        )

    def process(self, x):
        if self.max_length == -1:
            tokenized = self.tokenizer(
                x, return_tensors="pt", padding="longest"
            )
        else:
            tokenized = self.tokenizer(
                x,
                truncation=True,
                padding="longest",
                max_length=self.max_length,
                return_tensors="pt",
            )
        return tokenized

    def forward(self, x):
        tokenized = self.process(x)
        tokenized = tokenized.to(self.device)
        preds = torch.tensor([])
        preds = preds.to(self.device)

        encoding = self.model(**tokenized)
        # Retaining only the [CLS] token
        cls = encoding.last_hidden_state[:, 0, :]
        relu = nn.ReLU()
        # cls = relu(cls)
        # m = nn.Sigmoid()
        for label in self.labels:
            pred = self.prediction[label](cls)
            preds = torch.cat((preds, pred), dim=-1)

        # preds = F.normalize(preds, dim=0)

        # preds = relu(preds)
        # preds = m(preds)
        if self.mode == "train":
            return preds
        else:
            return preds, cls
