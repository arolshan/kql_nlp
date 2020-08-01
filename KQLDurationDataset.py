import torch
from torch.utils.data import Dataset

from Consts import *


class KQLDurationDataset(Dataset):
    def __init__(self, kql_queries, bucket_labels, raw_durations, container_labels, bucket_to_idx, container_to_idx, tokenizer, tokenizer_max_len):
        self.kql_queries = kql_queries
        self.bucket_labels = bucket_labels
        self.container_labels = container_labels
        self.raw_durations = raw_durations
        self.bucket_to_idx = bucket_to_idx
        self.container_to_idx = container_to_idx
        self.tokenizer = tokenizer
        self.tokenizer_max_len = tokenizer_max_len
        self.containers_onehot = torch.eye(len(container_to_idx), dtype=torch.float)

    def __len__(self):
        return len(self.kql_queries)

    def __getitem__(self, item):
        kql_query = f'[CLS] {str(self.kql_queries[item])}'
        raw_duration = self.raw_durations[item]
        bucket_idx = self.bucket_to_idx[self.bucket_labels[item]]
        container_idx = self.container_to_idx[self.container_labels[item]]
        encoding = self.tokenizer.encode(
          kql_query,
          add_special_tokens=True)
        encoding_ids = torch.tensor(encoding.ids, dtype=torch.long)
        attention_mask = (encoding_ids !=0).type(torch.int64)

        return {
            DL_KQL_QUERY: kql_query,
            DL_INPUT_IDS: encoding_ids,
            DL_ATTN_MASK: attention_mask,
            DL_BUCKETS: torch.tensor(bucket_idx, dtype=torch.long),
            DL_RAW_DURATIONS: torch.tensor(raw_duration, dtype=torch.float),
            DL_CONTAINERS: self.containers_onehot[container_idx]
        }
