#!/usr/bin/bash

python fed_train.py --total_size -1 --edge_fed_interval 1 --cloud_fed_interval 1 --pretrain_epochs 20 --pretrain_batch_cnt "20,20,20,20,20" --epochs_after_pretrain 20  #13
python fed_train.py --total_size -1 --edge_fed_interval 1 --cloud_fed_interval 1 --pretrain_epochs 20 --pretrain_batch_cnt "25,25,25,25,25" --epochs_after_pretrain 20  #14
python fed_train.py --total_size -1 --edge_fed_interval 1 --cloud_fed_interval 1 --pretrain_epochs 20 --pretrain_batch_cnt "30,30,30,30,30" --epochs_after_pretrain 20  #15
