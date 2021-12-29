import torch
from torch.utils.data import Dataset
from typing import Sequence, Tuple, List, Union
import linecache
import re
import os
import random
import subprocess
from Bio import SeqIO

class TrainerDataset(Dataset):
    def __init__(self, k=3):
        self.k = k

    def __len__(self):
        return 500000

    def __getitem__(self, index):
        rec = self.records[index]
        #return rec.id, re.sub('[(a-z)(\-)]', '', rec.seq.__str__())
        return rec.id, re.sub('[(\-)]', '', rec.seq.__str__())


class  MyDataset(Dataset):
    def __init__(self, names: List[str], lines: List[int]):
        self.names = names
        self.lines = lines

    def get_pair(self, path: str, lines: int) -> Tuple[str, str]:
        lines //= 2
        idx2 = random.randint(0, lines-1)
        seq1 = re.sub('[(\-)]', '', linecache.getline(path, 2))
        seq2 = re.sub('[(\-)]', '', linecache.getline(path, 2*idx2 + 2))

        return seq1, seq2

    def __getitem__(self, index: int) -> Tuple[str, str]:
        seq1, seq2 = self.get_pair(self.names[index], self.lines[index])
        return seq1, seq2

    def __len__(self):
        return len(self.names)

    def get_batch_indices(self, batch_size: int) -> List[List[int]]:
        batches = []
        buf = []
        iters = len(self.names) // batch_size

        for _ in range(iters):
            buf = random.sample(range(len(self.names)), batch_size)
            batches.append(buf)

        return batches


class SingleConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        limit_size = 500
        batch_size = len(raw_batch)
        max_len = max(len(seq) for id, seq in raw_batch)
        max_len = min(limit_size, max_len)
        ids = []
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)

        for i, (id, seq_str) in enumerate(raw_batch):
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq1 = torch.tensor([self.alphabet.get_idx(s) for s in seq_str[:limit_size]], dtype=torch.int64)
            ids.append(id)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : min(len(seq_str), max_len) + int(self.alphabet.prepend_bos),
            ] = seq1
            if self.alphabet.append_eos:
                tokens[i, min(len(seq_str), max_len) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
        return ids, tokens
