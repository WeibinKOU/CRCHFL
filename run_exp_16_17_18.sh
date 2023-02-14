#!/usr/bin/bash

python fed_train.py --total_size -1 --edge_fed_interval 1 --cloud_fed_interval 1 --pretrain_epochs 20 --pretrain_batch_cnt "15,15,15,15,15" --epochs_after_pretrain 25  #16
python fed_train.py --total_size -1 --edge_fed_interval 1 --cloud_fed_interval 1 --pretrain_epochs 20 --pretrain_batch_cnt "15,15,15,15,15" --epochs_after_pretrain 30  #17
python fed_train.py --total_size -1 --edge_fed_interval 1 --cloud_fed_interval 1 --pretrain_epochs 20 --pretrain_batch_cnt "15,15,15,15,15" --epochs_after_pretrain 35  #18
