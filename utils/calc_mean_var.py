import torch
from dataloader import ImgData
from config import data_transform
from tqdm import tqdm

def getStat():
    img_dataset = ImgData("../code/dataset/images/", None, data_transform)

    train_loader = torch.utils.data.DataLoader(
        img_dataset, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for _, X in tqdm(train_loader):
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(img_dataset))
    std.div_(len(img_dataset))
    return list(mean.numpy()), list(std.numpy())

if __name__ == '__main__':
    print(getStat())
