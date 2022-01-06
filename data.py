import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Sequence, Tuple, List, Union
import pandas as pd
import re
import random


class TrainerDataset(Dataset):
    def __init__(self, k=3, npair=62):
        self.k = k
        self.npair = npair

    def __len__(self):
        return 50

    def __getitem__(self, index):
        rand_set = torch.randint(0, 21, (self.npair*2, 3))
        qsample = rand_set[:self.npair]
        asample = rand_set[self.npair:]
        qv = F.one_hot(qsample, num_classes=21)
        av = qv.detach().clone()
        for i in range(self.npair):
            anchor = asample[i][0]%3
            newchar= asample[i][1]
            av[i][anchor] *= 0
            av[i][anchor][newchar] = 1
        return torch.cat((qv.permute(0,2,1), av.permute(0,2,1)), dim=0).float()
        

class EvalDataset(Dataset):
    def __init__(self, k=3, npair=62):
        self.k = k
        self.npair = npair

    def __len__(self):
        return 5000

    def __getitem__(self, index):
        
        qseed = torch.randint(0, 21, (2,3))
        nseed = torch.randint(1, 21, (self.npair+1, 3))
        qv = F.one_hot(qseed[0].unsqueeze(0), num_classes=21)
        a_ori = qseed[0].detach().clone()
        a_ori[qseed[1][0]%3] = (a_ori[qseed[1][0]%3]+nseed[0][0])%21
        av = F.one_hot(a_ori.unsqueeze(0), num_classes=21)
        n_ori = nseed[1:]
        n_ori += qseed[0].unsqueeze(0)
        n_ori %= 21
        nv = F.one_hot(n_ori, num_classes=21)
        res = torch.cat([qv, av, nv], dim=0)
        return res.permute(0,2,1).float()
        '''
        rand_set = torch.randint(0, 21, (self.npair + 2, 3))
        psample = rand_set[-2:]
        nsample = rand_set[:-2]
        qv = F.one_hot(psample[0], num_classes=21)
        pos = psample[1][0] % 3
        av = qv.detach().clone()
        av[pos] *= 0
        av[pos][psample[1][1]] = 1
        res = torch.stack([qv, av], dim=0)
        nv = F.one_hot(nsample, num_classes=21)
        bt = torch.cat((res, nv), dim=0)
        return bt.permute(0,2,1).float()
        '''
        

class FtDataset(Dataset):
    def __init__(self, k=3, npair=62):
        self.k = k
        self.npair = npair

    def __len__(self):
        return 50000

    def gen_diff2(self, base_vec, rand_seeds):
        mvec = base_vec.detach().clone()
        pivot1 = rand_seeds[0] % 3
        pivot2 = (pivot1 + 1) % 3
        mvec[pivot1] *= 0
        mvec[pivot2] *= 0
        mvec[pivot1][rand_seeds[1]] = 1
        mvec[pivot2][rand_seeds[2]] = 1
        return mvec

    def __getitem__(self, index):
        rand_set = torch.randint(0, 21, (self.npair + 2, 3))
        psample = rand_set[-2:]
        nsample = rand_set[:-2]
        qv = F.one_hot(psample[0], num_classes=21)
        pos = psample[1][0] % 3
        av = qv.detach().clone()
        av[pos] *= 0
        av[pos][psample[1][1]] = 1
        res = torch.stack([qv, av], dim=0)
        nv = torch.stack(
            [self.gen_diff2(qv, nsample[i]) for i in range(len(nsample))], dim=0
        )
        bt = torch.cat((res, nv), dim=0)
        return bt.float()


class OriginDataset(Dataset):
    def __init__(self, path):
        self.df = pd.read_table(path, sep=',', header=None)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        ret = self.df.iloc[index, :]
        return ret[0], ret[1]


alphabet = {
    'G': 0, 'A': 1, 'V': 2, 'L': 3, 'I': 4, 'F': 5, 'W': 6, 'Y': 7, 
    'D': 8, 'H': 9, 'N':10, 'E':11, 'K':12, 'Q':13, 'M':14, 'R':15, 
    'S':16, 'T':17, 'C':18, 'P':19, 'X':20, 'U':20, 'O':20

}

def translator(amino_str, alphabet):
    lst_str = list(amino_str)
    res = torch.tensor([alphabet[i] for i in lst_str])
    return F.one_hot(res, num_classes=21)

class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """
    def __init__(self, alphabet=alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch: Sequence[Tuple[int, str]]):
        lst_label, lst_seq = zip(*raw_batch)
        lst_label = list(lst_label)
        lst_seq = list(lst_seq)
        arg_idx = sorted(range(len(lst_seq)), key=lambda x: len(lst_seq[x]))
        lst_label = torch.tensor([lst_label[i] for i in arg_idx])
        lst_seq = [lst_seq[i] for i in arg_idx]
        lst_len = [len(lst_seq[i]) for i in arg_idx]
        lst_trans = [translator(i, self.alphabet) for i in lst_seq]
        data = pad_sequence(lst_trans, batch_first=True, padding_value=0)
        return lst_label, data.permute(0,2,1).float(), lst_len
