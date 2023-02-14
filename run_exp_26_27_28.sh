#!/usr/bin/bash

python fed_train.py --total_size 30 --edge_fed_interval 1 --cloud_fed_interval 1 --pretrain_epochs 20 --pretrain_batch_cnt "31,31,31,31,31" --epochs_after_pretrain 6  #26
python fed_train.py --total_size 35 --edge_fed_interval 1 --cloud_fed_interval 1 --pretrain_epochs 20 --pretrain_batch_cnt "39,39,39,39,39" --epochs_after_pretrain 6  #27
python fed_train.py --total_size 40 --edge_fed_interval 1 --cloud_fed_interval 1 --pretrain_epochs 20 --pretrain_batch_cnt "47,47,47,47,47" --epochs_after_pretrain 6  #28
