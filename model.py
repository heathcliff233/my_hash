import torch
import torch.nn as nn
import torch.nn.functional as F


class MyEncoder(nn.Module):
    def __init__(self, k=3, in_channels=21, out_channels=1000):
        super(MyEncoder, self).__init__()
        self.hash = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=k
        )

    def forward(self, x):
        x = self.hash(x)
        return x

    def get_loss(self, ebd):
        qebd = ebd[:, 0]
        qebd = qebd.reshape((qebd.shape[0], 1, qebd.shape[-1]))
        cebd = ebd[:, 1:]
        cebd = cebd.reshape((cebd.shape[0], cebd.shape[1], cebd.shape[2]))
        res = qebd * cebd
        sim_mx = res.sum(dim=-1, keepdim=False)
        label = torch.ones(sim_mx.shape[0], dtype=torch.long)
        sm_score = F.log_softmax(sim_mx, dim=1)
        loss = F.nll_loss(sm_score, label.to(sm_score.device), reduction="mean")
        return loss

    def get_acc(self, ebd):
        qebd = ebd[:, 0]
        qebd = qebd.reshape((qebd.shape[0], 1, qebd.shape[-1]))
        cebd = ebd[:, 1:]
        cebd = cebd.reshape((cebd.shape[0], cebd.shape[1], cebd.shape[2]))
        res = qebd * cebd
        sim_mx = res.sum(dim=-1, keepdim=False)
        label = torch.ones(sim_mx.shape[0], dtype=torch.long)
        sm_score = F.log_softmax(sim_mx, dim=1)
        max_score, max_idxs = torch.max(sm_score, 1)
        correct_predictions_count = (max_idxs == label.to(sm_score.device)).sum()
        return correct_predictions_count, sim_mx.shape[0]
