import torch
import torch.nn as nn
from transformers import LongformerModel, LongformerTokenizer

class BertBinary(nn.Module):

    def __init__(
        self,
        device,
        hidden_size=768,
        model_name="allenai/longformer-base-4096",
        truncation_side="right",
    ):
        super(BertBinary, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.model_name = model_name
        self.model = LongformerModel.from_pretrained(self.model_name)
        self.truncation_side = truncation_side
        self.tokenizer = LongformerTokenizer.from_pretrained(
            self.model_name, truncation_side=self.truncation_side
        )
        self.classification_layer = nn.Linear(self.hidden_size, 1)

    def process(self, x):
        tokenized = self.tokenizer(
            x,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        return tokenized

    def forward(self, x):
        tokenized = self.process(x)
        tokenized = tokenized.to(self.device)

        encoding = self.model(**tokenized)
        cls = encoding.last_hidden_state[:, 0, :]
        
        logits = self.classification_layer(cls)
        return logits.squeeze()
