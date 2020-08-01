import torch
import datetime
import logging
from torch import nn
from transformers import DistilBertModel, DistilBertConfig


class KQLDurationBucketClassifier(nn.Module):
    def __init__(self, n_classes, n_users, n_linear, vocab_size, h_sizes=[], drop_p = 0.1):
        super(KQLDurationBucketClassifier, self).__init__()
        assert n_linear >= 1
        config = DistilBertConfig()
        self.distilbert = DistilBertModel(config)
        self.distilbert.init_weights()
        self.distilbert.resize_token_embeddings(vocab_size)
        self.pre_classifier = nn.Linear(config.dim, config.dim)

        self.drop = nn.Dropout(p=drop_p)
        linear_sizes = [self.distilbert.config.hidden_size + n_users] + h_sizes[:]
        self.linears = nn.ModuleList()

        for k in range(len(linear_sizes) - 1):
            self.linears.append(
                nn.Sequential(
                    nn.Linear(linear_sizes[k], linear_sizes[k + 1]),
                    nn.ReLU(),
                    nn.Dropout(p=drop_p)))

        self.linears.append(nn.Linear(linear_sizes[-1], n_classes))

    def forward(self, input_ids, containers_onehot, attention_mask, print_info=False, batch_number=0):
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        dropout = self.drop(pooled_output)
        output = torch.cat([dropout, containers_onehot], dim=1)
        for layer in self.linears:
            output = layer(output)

        if (print_info):
            logging.info(f"Finished batch={batch_number}")
            logging.info(f"Finished batch={batch_number}")

        return output
