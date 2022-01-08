import wandb
import esm
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from model import MyEncoder
from data import TrainerDataset, EvalDataset
from train import train, evaluate

DISTRIBUTED = True
TRBATCHSZ = 1
EVBATCHSZ = 1
threshold = 0.7
eval_per_step = 10
lr = 1e-9
use_wandb = False


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
    model = MyEncoder(k=3, in_channels=21)
    device = torch.device("cuda:0")

    if use_wandb:
        init_wandb()

    model = model.to(device)
    # model.load_state_dict(torch.load("./saved_models/larger_batch/10.pth"))
    train_set = TrainerDataset(k=3, npair=128)
    eval_set = EvalDataset(k=3, npair=128)
    # train_set = FtDataset(k=3, npair=62)
    # eval_set = MyDataset(evnames, evlines)

    train_loader = DataLoader(dataset=train_set, batch_size=TRBATCHSZ, shuffle=False)
    eval_loader = DataLoader(dataset=eval_set, batch_size=TRBATCHSZ, shuffle=False)
    # eval_loader = DataLoader(dataset=eval_set, collate_fn=batch_converter, batch_sampler=evbatch)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)
    # print("res: ", evaluate(model, eval_loader))

    train(
        model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_epoches=60,
        optimizer=optimizer,
        threshold=threshold,
        eval_per_step=eval_per_step,
        use_wandb=use_wandb,
        device=device,
        acc_step=4,
    )
