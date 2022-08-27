import argparse
import numpy as np
import torch.nn as nn
import torch
import torchvision
from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import sys
import imgaug.augmenters as iaa
from config import *
from multistage_fed.fed_server import CloudServer
from multistage_fed.fed_scheduler import Scheduler

def print_device_info():
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
    parser.add_argument("--disable_pretrain", action='store_true', help="whether to enable pretaining stage to initialize all the models of edges and vehicles")
    parser.add_argument("--no_fl", action='store_true', help="whether to enable no-federated-learning")
    parser.add_argument("--pretrain_epochs", type=int, default=1, help="number of epochs of pretraining on Cloud Server")
    parser.add_argument("--pretrain_batch_cnt", type=int, default=5, help="how many batches of data that each vehicle should upload to Cloud Server to pretrain ")
    parser.add_argument("--edge_fed_interval", type=int, default=1, help="each edge_fed_interval vehicle training to do a Edge Server federated learning")
    parser.add_argument("--cloud_fed_interval", type=int, default=1, help="each cloud_fed_interval Edge Server federated learning to do a Cloud Server federated learning")
    parser.add_argument("--total_size", type=int, default=30, help="total size of communication resource")
    args = parser.parse_args()
    return args

def save_cmd(log_dir):
    txt_file = open(log_dir +'/cmd.txt', 'w')
    cmd=" ".join("\"" + arg + "\"" if " " in arg else arg for arg in sys.argv)
    cmd = 'python ' + cmd
    txt_file.write(cmd)
    txt_file.close()

def main():
    print_device_info()
    tb = SummaryWriter()
    save_cmd(tb.logdir)
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

    scheduler = Scheduler(args.total_size, 150, 450 * 4)

    scheduler.set_edge_fed_interval(args.edge_fed_interval)
    scheduler.set_cloud_fed_interval(args.cloud_fed_interval)
    scheduler.set_pretrain_epochs(args.pretrain_epochs)
    scheduler.set_pretrain_batch_cnt(args.pretrain_batch_cnt)

    platform = sys.platform
    if 'win' in platform:
        log_dir = tb.logdir.split('\\')[-1]
    elif 'linux' in platform:
        log_dir = tb.logdir.split('/')[-1]

    cloud = CloudServer(aug_seq, training_config, tb, scheduler, log_dir)

    try:
        if args.no_fl:
            cloud.CollectTrainData('Edge0')
            cloud.train()
            sys.exit()

        if not args.disable_pretrain:
            cloud.CollectPretrainData()
            cloud.Pretrain()
        cloud.SinkModelToEdges()
        for edge in cloud.edges:
            edge.SinkModelToClients()
        cloud.run()
    finally:
        tb.close()

if __name__ == '__main__':
    main()
