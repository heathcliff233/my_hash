import torch
from torch.utils.data import DataLoader
from model import MyEncoder, dot_product_scores
from data import OriginDataset, BatchConverter
from train import train, evaluate

TRBATCHSZ = 7
threshold = 0.7
eval_per_step = 10


if __name__ == "__main__":
    model = MyEncoder(k=3, in_channels=21)
    device = torch.device("cuda:0")

    model = model.to(device)
    model.load_state_dict(torch.load("./saved_models/cont/0.pth"))
    train_set = OriginDataset('./testset.csv')
    batch_converter = BatchConverter()
    train_loader = DataLoader(dataset=train_set, batch_size=TRBATCHSZ, shuffle=False, collate_fn=batch_converter)
    label, data, lens = iter(train_loader).next()
    res = model.gen_ebd(data.cuda(), lens)
    rate_mx = dot_product_scores(res, res).cpu() #/ torch.tensor(lens).float().unsqueeze(0)
    print(res.cpu())
    print(rate_mx)
    print(label)
    print(lens)

