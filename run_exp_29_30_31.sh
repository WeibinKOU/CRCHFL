#!/usr/bin/bash

python fed_train.py --total_size 20 --edge_fed_interval 1 --cloud_fed_interval 1 --pretrain_epochs 20 --pretrain_batch_cnt "36,36,36,36,36" --epochs_after_pretrain 0  #only_pretrain 29
python fed_train.py --total_size 20 --edge_fed_interval 1 --cloud_fed_interval 0 --pretrain_epochs 20 --pretrain_batch_cnt "42,42,0,0,0" --epochs_after_pretrain 18  #no_cloud 30
python fed_train.py --total_size 20 --edge_fed_interval 1 --cloud_fed_interval 1 --pretrain_epochs 0 --pretrain_batch_cnt "0,0,0,0,0" --epochs_after_pretrain 10  #no_pretrain 31
