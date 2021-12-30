import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Sequence, Tuple, List, Union
import linecache
import re
import random


class TrainerDataset(Dataset):
    def __init__(self, k=3):
        self.k = k
        self.npair = 10

    def __len__(self):
        return 500000

    def __getitem__(self, index):
        rand_set = torch.randint(0,21,(self.npair+2, 3))
        psample = rand_set[-2:]
        nsample = rand_set[:-2]
        qv = F.one_hot(psample[0],num_classes=21)
        pos = psample[1][0]%3
        av = qv.detach().clone()
        av[pos] *= 0
        av[pos][psample[1][1]] = 1
        res = torch.stack([qv, av], dim=0)
        nv = F.one_hot(nsample, num_classes=21)
        bt = torch.cat((res,nv), dim=0)
        return bt


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
                max_len
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)

        for i, (id, seq_str) in enumerate(raw_batch):
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq1 = torch.tensor(
                [self.alphabet.get_idx(s) for s in seq_str[:limit_size]],
                dtype=torch.int64,
            )
            ids.append(id)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : min(len(seq_str), max_len)
                + int(self.alphabet.prepend_bos),
            ] = seq1
            if self.alphabet.append_eos:
                tokens[
                    i, min(len(seq_str), max_len) + int(self.alphabet.prepend_bos)
                ] = self.alphabet.eos_idx
        return ids, tokens
