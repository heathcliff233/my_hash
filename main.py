import wandb
import esm
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from model import MyEncoder
from data import TrainerDataset
from train import train, evaluate

DISTRIBUTED = True
TRBATCHSZ = 64
EVBATCHSZ = 64
threshold = 0.7
eval_per_step = 30
lr = 1e-5
use_wandb = True


def init_wandb():
    wandb.init(
        project="cdhit",
        config={
            "optim": "AdamW",
            "lr": lr,
            "train_batch": TRBATCHSZ,
            "eval_per_step": EVBATCHSZ,
        },
    )


if __name__ == "__main__":
    model = MyEncoder(k=3, in_channels=21, out_channels=1000)
    device = torch.device("cuda:0")

    if use_wandb:
        init_wandb()

    model = model.to(device)
    train_set = TrainerDataset(k=3)
    # eval_set = MyDataset(evnames, evlines)

    train_loader = DataLoader(dataset=train_set, batch_size=TRBATCHSZ, shuffle=False)
    # eval_loader = DataLoader(dataset=eval_set, collate_fn=batch_converter, batch_sampler=evbatch)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.85)
    # print("res: ", evaluate(model, eval_loader))

    train(
        model,
        train_loader=train_loader,
        eval_loader=None,
        n_epoches=60,
        optimizer=optimizer,
        threshold=threshold,
        eval_per_step=eval_per_step,
        use_wandb=use_wandb,
        device=device,
        acc_step=4,
    )
