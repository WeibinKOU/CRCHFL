#!/usr/bin/bash

python fed_train.py --no_fl #1
python fed_train.py --total_size -1 --edge_fed_interval 1 --cloud_fed_interval -1 --pretrain_epochs 0 --pretrain_batch_cnt "0,0,0,0,0" --epochs_after_pretrain 20  #2
