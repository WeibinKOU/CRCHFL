import argparse
import numpy as np
import torch.nn as nn
import torch
import torchvision
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import imgaug.augmenters as iaa
from config import *
from multistage_fed.fed_server import CloudServer

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
    parser.add_argument("--enable_pretrain", type=bool, default=True, help="whether to enable pretaining stage to initialize all the models of edges and vehicles")
    args = parser.parse_args()
    return args

def main():
    tb = SummaryWriter()
    args = build_parser()

    training_config = {}
    training_config['epochs'] = args.epochs
    training_config['batch_size'] = args.batch_size
    training_config['lr'] = args.lr
    training_config['betas'] = (args.b1, args.b2)
    training_config['weight_decay'] = 1e-4

    aug_seq = iaa.Sequential([
        iaa.Affine(
            translate_percent={'x': (-.05, .05), 'y': (-.1, .1)},
            rotate=(-15, 15)
        ),

        iaa.GammaContrast((.4, 2.5)),
        iaa.GaussianBlur((0, 3.0)),

        iaa.Resize({'height': HEIGHT, 'width': WIDTH}),
    ])

    cloud = CloudServer(aug_seq, training_config, tb)

    try:
        if args.enable_pretrain:
            cloud.CollectPretrainData()
            cloud.Pretrain()
        cloud.SinkModelToEdges()
        for edge in cloud.edges:
            edge.SinkModelToClients()
        cloud.train()
    finally:
        tb.close()

if __name__ == '__main__':
    main()