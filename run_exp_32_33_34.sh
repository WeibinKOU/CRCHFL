#!/usr/bin/bash

python fed_train.py --total_size 20 --edge_fed_interval 2 --cloud_fed_interval 1 --pretrain_epochs 20 --pretrain_batch_cnt "10,10,10,10,10" --epochs_after_pretrain 8  #32
python fed_train.py --total_size 20 --edge_fed_interval 3 --cloud_fed_interval 1 --pretrain_epochs 20 --pretrain_batch_cnt "6,6,6,6,6" --epochs_after_pretrain 9  #33
python fed_train.py --total_size 20 --edge_fed_interval 4 --cloud_fed_interval 1 --pretrain_epochs 20 --pretrain_batch_cnt "4,4,4,4,4" --epochs_after_pretrain 12  #34
