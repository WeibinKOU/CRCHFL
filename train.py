import argparse
import os
import numpy as np
import math
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import imgaug as ia
import imgaug.augmenters as iaa

from fusion import ThrottleBrakePredModel, SteerPredModel
from dataloader import LidarData, ImgData, ActionData 
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
    parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--gpu", type=int, default=0, help="the index of GPU used to train")
    args = parser.parse_args()
    return args

def main():
    args = build_parser()

    bce_loss = nn.BCELoss().to(device)
    mse_loss = nn.MSELoss().to(device)
    huber_loss = nn.HuberLoss(delta=0.3).to(device)
    aug_seq = iaa.Sequential([
        #iaa.Fliplr(.5),

        iaa.Affine(
            translate_percent={'x': (-.05, .05), 'y': (-.1, .1)},
            rotate=(-25, 25)
        ),

        iaa.GammaContrast((.4, 2.5)),
        iaa.GaussianBlur((0, 3.0)),

        iaa.Resize({'height': 480, 'width': 640}),
    ])

    print("\nData loading ...\n")

    img_dataset = ImgData("../code/dataset/images/", aug_seq, data_transform)
    val_dataset = ImgData("../code/dataset/valid/images/", aug_seq, data_transform)
    lidar_dataset = LidarData("../code/dataset/lidars/")
    val_lidar_dataset = LidarData("../code/dataset/valid/lidars/")
    action_dataset = ActionData("../code/dataset/action.npy")

    #img_dataset = ImgData("./dataset/images/", aug_seq, data_transform)
    #val_dataset = ImgData("./dataset/valid/images/", aug_seq, data_transform)
    #lidar_dataset = LidarData("./dataset/lidars/")
    #val_lidar_dataset = LidarData("./dataset/valid/lidars/")
    #action_dataset = ActionData("./dataset/action.npy")

    img_dataloader = DataLoader(img_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True)

    val_dataloader = DataLoader(val_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True)

    tb_model = ThrottleBrakePredModel()
    st_model = SteerPredModel()
    tb_model.to(device)
    st_model.to(device)

    parameters = list(tb_model.parameters()) + list(st_model.parameters())
    optim = torch.optim.Adam(parameters, lr=args.lr, betas=(args.b1, args.b2), weight_decay=1e-4)

    for epoch in range(args.epochs):
        tb_model.train()
        st_model.train()

        avg_loss1_sum = 0.0
        avg_loss2_sum = 0.0
        batch_cnt = 0
        for name, imgs in tqdm(img_dataloader):
            action = action_dataset.getitems(name)[:, 0:3]
            #lidars = lidar_dataset.getitems(name)

            optim.zero_grad()

            imgs_cuda = imgs.to(device)
            shape = imgs_cuda.shape
            imgs_cuda_3d = imgs_cuda.reshape([shape[0] // THREED_CHANNEL, shape[1], THREED_CHANNEL, shape[2], shape[3]])

            output_tb = tb_model(imgs_cuda)
            output_st = st_model(imgs_cuda_3d)
            print(output_st)
            avg_loss1 = mse_loss(output_tb, action[:, 0:3:2].reshape([BATCH_SIZE, -1]))
            
            steer = (action[:, 1] + 1.0) / 2.0

            steer = steer.reshape([BATCH_SIZE // THREED_CHANNEL, THREED_CHANNEL, -1])
            avg_steer = torch.mean(steer, dim=1)
            avg_loss2 = mse_loss(output_st, avg_steer)

            avg_loss1.backward()
            avg_loss2.backward()
            optim.step()
            avg_loss1_sum += avg_loss1
            avg_loss2_sum += avg_loss2
            batch_cnt += 1

        loss1 = avg_loss1_sum / batch_cnt
        loss2 = avg_loss2_sum / batch_cnt
            
        log_info = "[Epoch: %d/%d] [Training Average Loss: %f, %f]" % (epoch, args.epochs, loss1.item(), loss2.item())
        print(log_info) 

        if False and epoch % 5 == 0:
            tb_model.eval()
            st_model.eval()
            val_batch_cnt = 0
            val_avg_loss1_sum = 0.0
            val_avg_loss2_sum = 0.0
            name_bk = None
            throttle, steer, brake, reverse = None, None, None, None
            for name, imgs in tqdm(val_dataloader):
                name_bk = name
                action = action_dataset.getitems(name)[:, 0:3]
                #lidars = val_lidar_dataset.getitems(name)

                imgs_cuda = imgs.to(device)
                shape = imgs_cuda.shape
                imgs_cuda_3d = imgs_cuda.reshape([shape[0] // THREED_CHANNEL, shape[1], THREED_CHANNEL, shape[2], shape[3]])

                output_tb = tb_model(imgs_cuda)
                output_st = st_model(imgs_cuda_3d)

                throttle, steer, brake = output_tb[0, 0], output_st[0, 0], output_tb[0, 1]

                avg_loss1 = mse_loss(output_tb, action[:, 0:3:2].reshape([BATCH_SIZE, -1]))
                avg_loss2 = mse_loss(output_st, action[:, 1].reshape([BATCH_SIZE, -1]))

                val_avg_loss1_sum += avg_loss1
                val_avg_loss2_sum += avg_loss2
                val_batch_cnt += 1

            val_loss1 = val_avg_loss1_sum / val_batch_cnt
            val_loss2 = val_avg_loss2_sum / val_batch_cnt

            throttle = throttle.detach().cpu().numpy()
            steer = steer.detach().cpu().numpy()
            brake = brake.detach().cpu().numpy()

            print("%s prediction action: " % name_bk[0], throttle, steer, brake)
            log_info = "[Epoch: %d/%d] [Validation Average Loss: %f, %f]" % (epoch, args.epochs, val_loss1.item(), val_loss2.item())
            print(log_info) 


        tb_name = "Epoch_%d_tb.pth" % (epoch)
        st_name = "Epoch_%d_st.pth" % (epoch)

        tb_save_path = os.path.join(ACTION_MODEL_PATH, tb_name) 
        st_save_path = os.path.join(ACTION_MODEL_PATH, st_name) 

        if epoch % 5 == 0:
            torch.save(tb_model.state_dict(), tb_save_path)
            torch.save(st_model.state_dict(), st_save_path)

if __name__ == "__main__":
    main()

