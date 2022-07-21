import argparse
import os
import numpy as np
import math
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

#from pointnet import PointNetDenseCls
#from resnet50 import ResNet50 
from fusion import ActionPredictModel
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

    # Init a dataloader object
    print("\nTraining data loading ...\n")

    img_dataset = ImgData("../code/dataset/images/", data_transform)
    #img_dataset = ImgData("./dataset/images/", data_transform)
    img_dataloader = DataLoader(img_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True)

    lidar_dataset = LidarData("../code/dataset/lidars/")
    action_dataset = ActionData("../code/dataset/action.npy")
    #lidar_dataset = LidarData("./dataset/lidars/")
    #action_dataset = ActionData("./dataset/action.npy")

    #resnet = ResNet50()
    #pointnet = PointNetDenseCls()

    #resnet.to(device)
    #pointnet.to(device)

    action_model = ActionPredictModel()
    action_model.to(device)

    parameters = action_model.parameters()
    optim = torch.optim.Adam(parameters, lr=args.lr, betas=(args.b1, args.b2))

    for epoch in range(args.epochs):
        #resnet.train()
        #pointnet.train()
        action_model.train()

        avg_loss_sum = 0.0
        batch_cnt = 1
        for name, imgs in tqdm(img_dataloader):
            torch.cuda.empty_cache()
            action = action_dataset.getitems(name)
            lidars = lidar_dataset.getitems(name)

            optim.zero_grad()

            '''
            img_throttle, img_steer, img_brake, img_reverse = resnet(imgs)
            throttle_loss1 = mse_loss(img_throttle, action[:, 0].reshape(BATCH_SIZE, -1)) 
            steer_loss1 = mse_loss(img_steer, action[:, 1].reshape(BATCH_SIZE, -1))
            brake_loss1 = mse_loss(img_brake, action[:, 2].reshape(BATCH_SIZE, -1))
            reverse_loss1 = bce_loss(img_reverse, action[:, 3].reshape(BATCH_SIZE, -1))
            img_avg_loss = (throttle_loss1 + steer_loss1 + brake_loss1 + reverse_loss1) / 4

            lidar_throttle, lidar_steer, lidar_brake, lidar_reverse = pointnet(lidars)
            throttle_loss2 = mse_loss(lidar_throttle, action[:, 0].reshape(BATCH_SIZE, -1)) 
            steer_loss2 = mse_loss(lidar_steer, action[:, 1].reshape(BATCH_SIZE, -1))
            brake_loss2 = mse_loss(lidar_brake, action[:, 2].reshape(BATCH_SIZE, -1))
            reverse_loss2 = bce_loss(lidar_reverse, action[:, 3].reshape(BATCH_SIZE, -1))
            lidar_avg_loss = (throttle_loss2 + steer_loss2 + brake_loss2 + reverse_loss2) / 4

            avg_loss = (img_avg_loss + lidar_avg_loss) / 2
            '''
            throttle, steer, brake, reverse = action_model(imgs, lidars)
            throttle_loss = mse_loss(throttle, action[:, 0].reshape(BATCH_SIZE, -1)) 
            steer_loss = mse_loss(steer, action[:, 1].reshape(BATCH_SIZE, -1))
            brake_loss = bce_loss(brake, action[:, 2].reshape(BATCH_SIZE, -1))
            reverse_loss = bce_loss(reverse, action[:, 3].reshape(BATCH_SIZE, -1))
            avg_loss = (1.5 * throttle_loss + 1.5 * steer_loss + 0.5 * brake_loss + 0.5 * reverse_loss) / 4.0
            #print(avg_loss)

            #avg_loss.backward(retain_graph=True)
            avg_loss.backward()
            optim.step()
            avg_loss_sum += avg_loss
            batch_cnt += 1

        #resnet.eval()
        #pointnet.eval()

        #valid_feat = encoder(valid_dataset)
        #valid_output = decoder(valid_feat)
        #valid_loss = mse_loss(valid_dataset, valid_output)
        loss = avg_loss_sum / batch_cnt
            
        log_info = "[Epoch: %d/%d] [Training Average Loss: %f]" % (epoch, args.epochs, loss.item())
        print(log_info) 

        #resnet_name = "Epoch_%d_resnet.pth" % (epoch)
        #pointnet_name = "Epoch_%d_pointnet.pth" % (epoch)
        actionM_name = "Epoch_%d_action.pth" % (epoch)

        #resnet_save_path = os.path.join(RESNET_MODEL_PATH, resnet_name)
        #pointnet_save_path = os.path.join(POINTNET_MODEL_PATH, pointnet_name) 
        actionM_save_path = os.path.join(ACTION_MODEL_PATH, actionM_name) 

        if epoch % 5 == 0:
            #torch.save(resnet.state_dict(), resnet_save_path)
            #torch.save(pointnet.state_dict(), pointnet_save_path)
            torch.save(action_model.state_dict(), actionM_save_path)

if __name__ == "__main__":
    main()

