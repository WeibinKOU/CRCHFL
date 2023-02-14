#!/usr/bin/bash

python fed_train.py --total_size 20 --edge_fed_interval 1 --cloud_fed_interval 1 --pretrain_epochs 20 --pretrain_batch_cnt "15,15,15,15,15" --epochs_after_pretrain 6  #proposed 20
python fed_train.py --total_size 20 --edge_fed_interval 1 --cloud_fed_interval 1 --pretrain_epochs 25 --pretrain_batch_cnt "15,15,15,15,15" --epochs_after_pretrain 6  #21
python fed_train.py --total_size 20 --edge_fed_interval 1 --cloud_fed_interval 1 --pretrain_epochs 30 --pretrain_batch_cnt "15,15,15,15,15" --epochs_after_pretrain 6  #22
