Imitation-Learning-to-Drive-the-Vehicle
===
This repository aims to build a multi-stage framework to train and evaluate a Federated Imitation Learning(FIL) model to control vehicle automatically.<br>
There are two files which can be used to do this task, named train.py and fed\_train.py respectively. File train.py is mainly used to train a model that<br>
the training dataset is located at a single location, while the fed\_train.py is used to train the model with distributed dataset.<br>

Here I mainly introduce how to use fed\_train.py with different cases: <br>

1. No FL: <br>
---
Although it is originally desigend for federated learning framework, it can be used to train a model without FL. Now you should change the file<br>
multistage\_fed/fed\_config.py to config the dataset location. You can manually to add or delete items in Edge0(default). The cmd to train is:<br>
python fed\_train.py --no\_fl <br>

2. Unlimited data: <br>
---
In this case, the data throughput is unlimited, in other word, you can transfer training data and models in the process of<br>
training the FIL model without any limitation. And it do edge federated learning and cloud federated learning at every epoch.The cmd to train is:<br>
python fed\_train.py --disable\_pretrain --total\_size -1 --edge\_fed\_interval 1 --cloud\_fed\_interval 1 <br>

3. Limited data & Disable pretrain: <br>
---
In this case, the data for FL is limited and we train the model withou the pretrain stage, which does not consume data so that it can train more epochs.<br>
And we still do the edge federated learning and cloud federated learning at every epoch. The cmd to train is: <br>
python fed\_train.py --disable\_pretrain --total\_size 30 --edge\_fed\_interval 1 --cloud\_fed\_interval 1 <br>

4. Limited data & Enable pretrain: <br>
---
In this case, the data for FL is limited and we enable the pretrain stage to train a model. Owing to transfer some training data to cloud server, so less <br>
epochs is for the training. We still do the edge federated learning and cloud federated learning at every epoch in this case. The cmd to train is: <br>
python fed\_train.py --total\_size 30 --edge\_fed\_interval 1 --cloud\_fed\_interval 1 --pretrain\_epochs 20 --pretrain\_batch\_cnt 50 <br>

5. Limited data & Enable pretrain & Optimized fed interval: <br>
---
Compared with the previous case, in this case we just change the edge federated learning interval and cloud federated learning interval. The cmd is: <br>
python fed\_train.py --total\_size 30 --edge\_fed\_interval 2 --cloud\_fed\_interval 2 --pretrain\_epochs 20 --pretrain\_batch\_cnt 50 <br>

6. Limited data & Enable pretrain with more batches of training data & Optimized fed interval: <br>
---
Compared with the previous case, this case we just change the upload more data to cloud server to do pretrain. The cmd is: <br>
python fed\_train.py --total\_size 30 --edge\_fed\_interval 2 --cloud\_fed\_interval 2 --pretrain\_epochs 20 --pretrain\_batch\_cnt 60 <br>
