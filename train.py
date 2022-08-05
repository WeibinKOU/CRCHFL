import argparse
import os
import numpy as np
import math
import torch.nn as nn
import torch
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import imgaug as ia
import imgaug.augmenters as iaa
from utils.one_hot import label2onehot

from fusion import ThrottlePredModel, BrakePredModel, SteerPredModel
from dataloader import LidarData, ImgData, ActionData, MultiImgData
from config import *

np.random.seed(0)
torch.manual_seed(0)
print(torch.__version__)


if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.enabled = False 
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(torch.cuda.get_device_name(0))

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.003, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--gpu", type=int, default=0, help="the index of GPU used to train")
    args = parser.parse_args()
    return args

def main():
    tb = SummaryWriter()
    args = build_parser()

    mse_loss = nn.MSELoss().to(device)
    l1_loss = nn.L1Loss().to(device)
    mce_loss = nn.CrossEntropyLoss().to(device)
    bce_loss = nn.BCELoss().to(device)

    softmax = nn.Softmax(dim=1)

    aug_seq = iaa.Sequential([
        #iaa.Fliplr(.5),

        iaa.Affine(
            translate_percent={'x': (-.05, .05), 'y': (-.1, .1)},
            rotate=(-15, 15)
        ),

        iaa.GammaContrast((.4, 2.5)),
        iaa.GaussianBlur((0, 3.0)),

        iaa.Resize({'height': HEIGHT, 'width': WIDTH}),
    ])

    print("\nData loading ...\n")

    img_dataset = MultiImgData("./dataset/adj_pitch/balanced/", None, data_transform)
    train_action = ActionData("./dataset/adj_pitch/balanced/cla_action.npy")

    img_dataloader = DataLoader(img_dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True)

    data_len = len(img_dataloader) * BATCH_SIZE
    steer_model = SteerPredModel()

    steer_model.to(device)

    parameters = steer_model.parameters()
    optim = torch.optim.Adam(parameters, lr=args.lr, betas=(args.b1, args.b2), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.1)

    images_f = torch.rand([BATCH_SIZE, 3, HEIGHT, WIDTH], dtype=torch.float32).to(device)
    images_l = torch.rand([BATCH_SIZE, 3, HEIGHT, WIDTH], dtype=torch.float32).to(device)
    images_r = torch.rand([BATCH_SIZE, 3, HEIGHT, WIDTH], dtype=torch.float32).to(device)

    tb.add_graph(steer_model, (images_f, images_l, images_r))

    batch_cnt = len(img_dataloader)
    for epoch in range(args.epochs):
        steer_model.train()

        avg_loss2_sum = 0.0
        right_cnt = 0
        for name, imgs_f, imgs_l, imgs_r in tqdm(img_dataloader):
            action = train_action.getitems(name)[:, 0:3]

            optim.zero_grad()

            imgs_f = imgs_f.to(device)
            imgs_l = imgs_l.to(device)
            imgs_r = imgs_r.to(device)

            output_steer, feat = steer_model(imgs_f, imgs_l, imgs_r)
            out_steer = output_steer.clone()

            steer = action[:, 1].reshape([BATCH_SIZE, -1])
            label = np.array(steer.squeeze())

            steer = Tensor(label2onehot(steer, 3))
            avg_loss2 = mce_loss(output_steer, steer)

            avg_loss2.backward()
            optim.step()

            avg_loss2_sum += avg_loss2

            out_prob = softmax(out_steer)
            out_prob = np.argmax(out_prob.detach().cpu().numpy(), axis=1)
            pred_res = out_prob==label
            right_cnt += sum(pred_res)
            print(pred_res)
            print("Matching No.: ", right_cnt)

            #tb.add_embedding(mat=feat, metadata=label, label_img=imgs_f, global_step=30)

        scheduler.step()

        loss2 = avg_loss2_sum / batch_cnt
        acc = right_cnt / data_len

        log_info = "[Epoch: %d/%d] [Training Average Loss: %f, Accuracy: %f]" % (epoch, args.epochs, loss2.item(), acc)
        print(log_info)

        tb.add_scalar('Loss', loss2, epoch)
        tb.add_scalar('Accuracy', acc, epoch)

        tb.add_histogram('img_f.conv1.weight', steer_model.img_f.conv1.weight, epoch)
        tb.add_histogram('img_f.conv1.bias', steer_model.img_f.conv1.bias, epoch)
        tb.add_histogram('img_f.conv1.weight.grad' ,steer_model.img_f.conv1.weight.grad ,epoch)

        tb.add_histogram('img_l.conv1.weight', steer_model.img_l.conv1.weight, epoch)
        tb.add_histogram('img_l.conv1.bias', steer_model.img_f.conv1.bias, epoch)
        tb.add_histogram('img_l.conv1.weight.grad' ,steer_model.img_l.conv1.weight.grad ,epoch)

        tb.add_histogram('img_r.conv1.weight', steer_model.img_r.conv1.weight, epoch)
        tb.add_histogram('img_r.conv1.bias', steer_model.img_r.conv1.bias, epoch)
        tb.add_histogram('img_r.conv1.weight.grad' ,steer_model.img_r.conv1.weight.grad ,epoch)

        tb.add_histogram('fc.linear1.weight', steer_model.fc.linear1.weight, epoch)
        tb.add_histogram('fc.linear1.bias', steer_model.fc.linear1.bias, epoch)
        tb.add_histogram('fc.linear1.weight.grad' ,steer_model.fc.linear1.weight.grad ,epoch)

        steer_name = "Epoch_%d_steer.pth" % (epoch)

        steer_save_path = os.path.join(ACTION_MODEL_PATH, steer_name) 

        if epoch % 3 == 0:
            torch.save(steer_model.state_dict(), steer_save_path)

    tb.close()

if __name__ == "__main__":
    main()
