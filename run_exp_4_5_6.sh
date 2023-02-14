#!/usr/bin/bash

python fed_train.py --total_size -1 --edge_fed_interval 2 --cloud_fed_interval 1 --pretrain_epochs 20 --pretrain_batch_cnt "15,15,15,15,15" --epochs_after_pretrain 20  #4
python fed_train.py --total_size -1 --edge_fed_interval 3 --cloud_fed_interval 1 --pretrain_epochs 20 --pretrain_batch_cnt "15,15,15,15,15" --epochs_after_pretrain 20  #5
python fed_train.py --total_size -1 --edge_fed_interval 4 --cloud_fed_interval 1 --pretrain_epochs 20 --pretrain_batch_cnt "15,15,15,15,15" --epochs_after_pretrain 20  #6
