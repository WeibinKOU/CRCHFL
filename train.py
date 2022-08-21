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

from fusion import ActionPredModel
from dataloader import ImgData, ActionData, MultiImgData
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

def Evaluate(model, eval_data, mse_loss, mce_loss, softmax, batch_size):
    model.eval()

    dataloader, action, data_len, batch_cnt = eval_data

    right_cnt = 0
    avg_loss1_sum = 0.0
    avg_loss2_sum = 0.0
    loss = 0.0
    steer_acc = 0.0

    with torch.no_grad():
        for names, imgs_f, imgs_l, imgs_r, imgs_b in tqdm(dataloader):
            action_batch = action.getitems(names)
            st_action, tb_action = action_batch[:, 1], action_batch[:, 0:3:2]

            imgs_f = imgs_f.to(device)
            imgs_l = imgs_l.to(device)
            imgs_r = imgs_r.to(device)
            imgs_b = imgs_b.to(device)

            out_thro_brake, out_steer = model(imgs_f, imgs_l, imgs_r, imgs_b)

            thro_brake = tb_action.reshape([batch_size, -1])
            avg_loss1 = mse_loss(out_thro_brake, Tensor(thro_brake))

            steer = st_action.reshape([batch_size, -1])
            label = steer.squeeze()
            steer = Tensor(label2onehot(steer, 7))
            avg_loss2 = mce_loss(out_steer, steer)

            avg_loss1_sum += avg_loss1
            avg_loss2_sum += avg_loss2

            out_prob = softmax(out_steer)

            out_prob = np.argmax(out_prob.detach().cpu().numpy(), axis=1)

            pred_res = out_prob==np.array(label)
            right_cnt += sum(pred_res)

        loss1 = avg_loss1_sum / batch_cnt
        loss2 = avg_loss2_sum / batch_cnt
        loss = loss1.item() + loss2.item()

        steer_acc = right_cnt / data_len

    return loss, steer_acc


def main():
    tb = SummaryWriter()
    args = build_parser()

    mse_loss = nn.MSELoss().to(device)
    l1_loss = nn.L1Loss().to(device)
    mce_loss = nn.CrossEntropyLoss().to(device)
    bce_loss = nn.BCELoss().to(device)

    softmax = nn.Softmax(dim=1)

    aug_seq = iaa.Sequential([
        iaa.Affine(
            translate_percent={'x': (-.05, .05), 'y': (-.1, .1)},
            rotate=(-15, 15)
        ),

        iaa.GammaContrast((.4, 2.5)),
        iaa.GaussianBlur((0, 3.0)),

        iaa.Resize({'height': HEIGHT, 'width': WIDTH}),
    ])

    print("\nData loading ...\n")

    dataset = MultiImgData("./dataset/town01/", aug_seq, data_transform)
    action = ActionData("./dataset/town01/cla7_action.npy")

    eval_dataset = MultiImgData("./dataset/test/", aug_seq, data_transform)
    eval_action = ActionData("./dataset/test/cla7_action.npy")

    dataloader = DataLoader(dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True)
    data_len = len(dataloader) * args.batch_size

    eval_dataloader = DataLoader(eval_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True)
    eval_data_len = len(eval_dataloader) * args.batch_size
    eval_data = [eval_dataloader, eval_action, eval_data_len, len(eval_dataloader)]

    action_model = ActionPredModel()

    action_model.to(device)

    parameters = action_model.parameters()
    optim = torch.optim.Adam(parameters, lr=args.lr, betas=(args.b1, args.b2), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.1)

    images_f = torch.rand([args.batch_size, 3, HEIGHT, WIDTH], dtype=torch.float32).to(device)
    images_l = torch.rand([args.batch_size, 3, HEIGHT, WIDTH], dtype=torch.float32).to(device)
    images_r = torch.rand([args.batch_size, 3, HEIGHT, WIDTH], dtype=torch.float32).to(device)
    images_b = torch.rand([args.batch_size, 3, HEIGHT, WIDTH], dtype=torch.float32).to(device)

    tb.add_graph(action_model, (images_f, images_l, images_r, images_b))

    batch_cnt = len(dataloader)
    for epoch in range(args.epochs):
        action_model.train()

        avg_loss1_sum = 0.0
        avg_loss2_sum = 0.0
        right_cnt = 0
        for names, imgs_f, imgs_l, imgs_r, imgs_b in tqdm(dataloader):
            actions = action.getitems(names)

            st_action = actions[:, 1]
            tb_action = actions[:, 0:3:2]

            optim.zero_grad()

            imgs_f = imgs_f.to(device)
            imgs_l = imgs_l.to(device)
            imgs_r = imgs_r.to(device)
            imgs_b = imgs_b.to(device)

            out_thro_brake, out_steer = action_model(imgs_f, imgs_l, imgs_r, imgs_b)

            thro_brake = tb_action.reshape([args.batch_size, -1])
            avg_loss1 = mse_loss(out_thro_brake, Tensor(thro_brake))

            steer = st_action.reshape([args.batch_size, -1])
            label = steer.squeeze()
            steer = Tensor(label2onehot(steer, 7))
            avg_loss2 = mce_loss(out_steer, steer)

            avg_loss1.backward()
            avg_loss2.backward()

            optim.step()

            avg_loss1_sum += avg_loss1
            avg_loss2_sum += avg_loss2

            out_prob = softmax(out_steer)

            out_prob = np.argmax(out_prob.detach().cpu().numpy(), axis=1)

            pred_res = out_prob==np.array(label)
            right_cnt += sum(pred_res)

        scheduler.step()

        loss1 = avg_loss1_sum / batch_cnt
        loss2 = avg_loss2_sum / batch_cnt

        loss = loss1.item() + loss2.item()
        acc = right_cnt / data_len

        log_info = "[Epoch: %d/%d] [Train.Loss: %f, Train.Accuracy: %f]" % (epoch, args.epochs, loss, acc)
        print(log_info)

        tb.add_scalar('Train.Loss', loss, epoch)
        tb.add_scalar('Train.Accuracy', acc, epoch)

        eval_loss, eval_acc = Evaluate(action_model, eval_data, mse_loss, mce_loss, softmax, args.batch_size)
        tb.add_scalar('Eval.Loss', eval_loss, epoch)
        tb.add_scalar('Eval.Accuracy', eval_acc, epoch)
        log_info = "[Epoch: %d/%d] [Eval.Loss: %f, Eval.Accuracy: %f]" % (epoch, args.epochs, eval_loss, eval_acc)
        print(log_info)

        if epoch % 3 == 0:
            action_name = "Epoch_%d_action.pth" % (epoch)
            action_save_path = os.path.join(ACTION_MODEL_PATH, action_name)
            torch.save(action_model.state_dict(), action_save_path)

    tb.close()

if __name__ == "__main__":
    main()
