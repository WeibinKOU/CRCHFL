#!/usr/bin/bash

python fed_train.py --total_size 20 --edge_fed_interval 1 --cloud_fed_interval 1 --pretrain_epochs 35 --pretrain_batch_cnt "15,15,15,15,15" --epochs_after_pretrain 6  #23
python fed_train.py --total_size 20 --edge_fed_interval 1 --cloud_fed_interval 1 --pretrain_epochs 40 --pretrain_batch_cnt "15,15,15,15,15" --epochs_after_pretrain 6  #24
python fed_train.py --total_size 25 --edge_fed_interval 1 --cloud_fed_interval 1 --pretrain_epochs 20 --pretrain_batch_cnt "23,23,23,23,23" --epochs_after_pretrain 6  #25
