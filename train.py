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

    steer_dataset = MultiImgData("./dataset/adj_pitch_balanced/", aug_seq, data_transform)
    steer_action = ActionData("./dataset/adj_pitch_balanced/cla7_action.npy")

    thro_brake_dataset = ImgData("../code/dataset/images/", aug_seq, data_transform)
    thro_brake_action = ActionData("../code/dataset/action.npy")

    steer_dl = DataLoader(steer_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True)

    thro_brake_dl = DataLoader(thro_brake_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True)

    data_len = len(steer_dl) * BATCH_SIZE
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

    batch_cnt = min(len(steer_dl), len(thro_brake_dl))
    for epoch in range(args.epochs):
        action_model.train()

        avg_loss1_sum = 0.0
        avg_loss2_sum = 0.0
        right_cnt = 0
        for i in tqdm(range(batch_cnt)):
            params1, params2 = next(iter(steer_dl)), next(iter(thro_brake_dl))
            names1, imgs_f, imgs_l, imgs_r = params1[0], params1[1], params1[2], params1[3]
            names2, imgs_b = params2[0], params2[1]

            st_action = steer_action.getitems(names1)[:, 1]
            tb_action = thro_brake_action.getitems(names2)[:, 0:3:2]

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
            print(out_prob)

            out_prob = np.argmax(out_prob.detach().cpu().numpy(), axis=1)

            pred_res = out_prob==np.array(label)
            right_cnt += sum(pred_res)

            print("Matching No.: ", sum(pred_res))

            #tb.add_embedding(mat=feat, metadata=label, label_img=imgs_f, global_step=30)

        scheduler.step()

        loss1 = avg_loss1_sum / batch_cnt
        loss2 = avg_loss2_sum / batch_cnt

        acc = right_cnt / data_len

        log_info = "[Epoch: %d/%d] [Throttle and brake Loss: %f, Steer Loss: %f, Steer accuracy: %f]" % (epoch, args.epochs, loss1.item(), loss2.item(), acc)
        print(log_info)

        tb.add_scalar('Thro_brake Loss', loss1, epoch)
        tb.add_scalar('Steer Loss', loss2, epoch)
        tb.add_scalar('Steer Accuracy', acc, epoch)

        #tb.add_histogram('extractor.conv1.weight', steer_model.extractor.conv1.weight, epoch)
        #tb.add_histogram('extractor.conv1.bias', steer_model.extractor.conv1.bias, epoch)
        #tb.add_histogram('extractor.conv1.weight.grad' ,steer_model.extractor.conv1.weight.grad ,epoch)

        action_name = "Epoch_%d_action.pth" % (epoch)

        action_save_path = os.path.join(ACTION_MODEL_PATH, action_name)

        if epoch % 3 == 0:
            torch.save(action_model.state_dict(), action_save_path)

    tb.close()

if __name__ == "__main__":
    main()
