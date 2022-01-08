import wandb
import pickle
import torch


def train(
    model,
    train_loader,
    eval_loader,
    n_epoches,
    optimizer,
    threshold=0.7,
    eval_per_step=10,
    use_wandb=False,
    device="cuda:0",
    acc_step=1,
):
    if use_wandb:
        wandb.watch(model, log_freq=eval_per_step)
    for epoch in range(n_epoches):
        print("epoch " + str(epoch + 1))
        tot_loss = 0

        model.train()

        for cnt, toks in enumerate(train_loader, start=1):
            toks = toks.squeeze(0)
            toks = toks.cuda(non_blocking=True)

            if cnt % eval_per_step == 0:
                if cnt % (eval_per_step * 1) == 0:
                    # acc = evaluate(model, eval_loader, threshold)
                    ac2 = evaluate(model, eval_loader, threshold)

                    # acc = acc.view(-1).cpu().item()
                    # print("acc: ", acc)
                    print("loss" + str(tot_loss / eval_per_step))
                    print("acc: %.8f" % ac2.cpu().item())
                    if use_wandb:
                        wandb.log(
                            {"train/train-acc": ac2}
                        )  # , "train/eval-acc": acc,"train/loss": tot_loss/eval_per_step})
                    tot_loss = 0
                model.train()

            if cnt % acc_step == 0:
                optimizer.zero_grad()
                loss = model.get_loss(model(toks))
                tot_loss += loss.detach().cpu().item()
                loss.backward()
                optimizer.step()
            else:
                loss = model.get_loss(model(toks))
                loss.backward()
        save(model, epoch)


def evaluate(model, loader, threshold=0.7):
    model.eval()
    correct = torch.tensor([0]).cuda()
    total = torch.tensor([0]).cuda()
    for i, toks in enumerate(loader):
        toks = toks.squeeze(0)
        toks = toks.cuda(non_blocking=True)
        if i > 4000:
            break
        right, num = model.get_real_acc(model(toks))
        correct += right
        total += num
    return correct / total


def do_embedding(model, loader, path, device="cuda:0"):
    model.eval()
    res = []
    for i, (ids, toks) in enumerate(loader):
        toks = toks.squeeze(0)
        toks.transpose_(1, 2)
        toks = toks.to(device)
        with torch.no_grad():
            out = model.forward_once(toks)
        out = out.cpu().numpy()
        res.extend([(ids[i], out[i]) for i in range(out.shape[0])])
    with open(path[:-4], mode="wb") as f:
        pickle.dump(res, f)


def save(model, epoch):
    torch.save(model.state_dict(), "./saved_models/cont/" + str(epoch) + ".pth")
