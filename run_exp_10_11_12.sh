#!/usr/bin/bash

python fed_train.py --total_size -1 --edge_fed_interval 1 --cloud_fed_interval 1 --pretrain_epochs 25 --pretrain_batch_cnt "15,15,15,15,15" --epochs_after_pretrain 20  #10
python fed_train.py --total_size -1 --edge_fed_interval 1 --cloud_fed_interval 1 --pretrain_epochs 30 --pretrain_batch_cnt "15,15,15,15,15" --epochs_after_pretrain 20  #11
python fed_train.py --total_size -1 --edge_fed_interval 1 --cloud_fed_interval 1 --pretrain_epochs 35 --pretrain_batch_cnt "15,15,15,15,15" --epochs_after_pretrain 20  #12
