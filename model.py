import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def dot_product_scores(q_vectors, ctx_vectors):
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r

def residule_score(q_vectors, ctx_vectors):
    """
    calculate distance between each vector in q and ctx
    """
    res = q_vectors.unsqueeze(1) - ctx_vectors.unsqueeze(0)
    return -1.0 * (res**2).sum(-1)

class MyEncoder(nn.Module):
    def __init__(self, k=3, in_channels=21, out_channels=21**3):
        super(MyEncoder, self).__init__()
        self.hash1 = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=k
        )
        self.hash2 = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=k
        )
        self.hash3 = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=k
        )

    def forward(self, x):
        x1 = self.hash1(x)
        x2 = self.hash2(x)
        x3 = self.hash3(x)
        x = x1 * x2 * x3
        
        #eps = 1e-3
        #holder = torch.cuda.FloatTensor(x.size())
        #torch.randn(x.size(), out=holder)
        #x += eps * holder
        
        x = F.normalize(x, p=2, dim=1)
        return x

    def get_loss(self, ebd):
        ebd = ebd.squeeze(2)
        tot_len = len(ebd)
        qebd = ebd[:tot_len//2]
        cebd = ebd[tot_len//2:]
        sim_mx = dot_product_scores(qebd, cebd)
        label = torch.arange(sim_mx.shape[0], dtype=torch.long)
        tau = 0.07
        sm_score = F.log_softmax(sim_mx/tau, dim=1)
        loss1 = F.nll_loss(
            sm_score,
            label.to(sm_score.device),
            reduction="mean"
        )
        #mask = torch.ones(sim_mx.shape) - torch.eye(sim_mx.shape[0])
        #out = mask.cuda() * sim_mx
        loss2 = sim_mx.pow(2).exp().mean()
        #dist_mx = residule_score(qebd, cebd)
        #ds_score = F.log_softmax(dist_mx/tau, dim=1)
        #loss2 = F.nll_loss(
        #    dist_mx,
        #    label.to(ds_score.device),
        #    reduction="mean"
        #)
        return loss2 #+ loss1
        '''
        qebd = ebd[0]
        qebd = qebd.reshape((1, -1))
        cebd = ebd[1:]
        cebd = cebd.reshape((cebd.shape[0], -1))
        res = qebd * cebd
        sim_mx = res.sum(dim=-1, keepdim=True).reshape((1,-1))
        tau = 0.07
        loss = -1.0 * F.log_softmax(sim_mx / tau, dim=-1)[0][0]
        #label = torch.ones(1, dtype=torch.long)
        #sm_score = F.log_softmax(sim_mx, dim=-1)
        #loss = F.nll_loss(sm_score, label.to(sm_score.device), reduction="mean")
        return loss
        '''

    def get_acc(self, ebd):
        ebd = ebd.squeeze(2)
        tot_len = len(ebd)
        qebd = ebd[:tot_len//2]
        cebd = ebd[tot_len//2:] 
        sim_mx = dot_product_scores(qebd, cebd)
        label = torch.arange(sim_mx.shape[0], dtype=torch.long)
        sm_score = F.log_softmax(sim_mx, dim=1)
        max_score, max_idxs = torch.max(sm_score, 1)
        correct_predictions_count = (
            max_idxs == label.to(sm_score.device)
        ).sum()
        return correct_predictions_count, sim_mx.shape[0]
    
    def get_real_acc(self, ebd):
        qebd = ebd[0]
        qebd = qebd.reshape((1, -1))
        cebd = ebd[1:]
        cebd = cebd.reshape((cebd.shape[0], -1))
        res = qebd * cebd
        sim_mx = res.sum(dim=-1, keepdim=False).reshape((1,-1))
        label = torch.zeros(1, dtype=torch.long)
        sm_score = F.log_softmax(sim_mx, dim=-1)
        max_score, max_idxs = torch.max(sm_score, 1)
        correct_predictions_count = (max_idxs == label.to(sm_score.device)).sum()
        return correct_predictions_count, 1

    def gen_ebd(self, data, lens):
        with torch.no_grad():
            data = self.forward(data)
            res = torch.stack([data[i][:lens[i]-1].sum(dim=0) for i in range(len(lens))])
        return res
        
