Federated-Imitation-Learning-to-Drive-the-Vehicle
===
This repository aims to build a multi-stage framework to train and evaluate a Federated Imitation Learning(FIL) model to control vehicle automatically.
There are two files which can be used to do this task, named train.py and fed\_train.py respectively. File train.py is mainly used to train a model that
the training dataset is located at a single location, while the fed\_train.py is used to train the model with distributed dataset.

Here I mainly introduce how to use fed\_train.py with different cases:

1. No FL:
---
Although it is originally desigend for federated learning framework, it can be used to train a model without FL. Now you should change the config file
multistage\_fed/fed\_config.py to setup the dataset location. You can manually add or delete items in Edge0(default) based on your real case. The command to train is:<br>
```bash
python fed\_train.py --no\_fl
```

2. Unlimited data & Disable pretrain:
---
In this case, the data throughput is unlimited. In other words, you can transfer training data and models in the process of
training the FIL model without any limitation. And it does edge federated learning and cloud federated learning at every epoch.The command to train is:<br>
```bash
python fed\_train.py --disable\_pretrain --total\_size -1 --edge\_fed\_interval 1 --cloud\_fed\_interval 1
```

3. Limited data & Disable pretrain:
---
In this case, the data for FL is limited and it trains the model without the pretraining stage, which does not consume data so that it can train more epochs.
And it still does the edge federated learning and cloud federated learning at every epoch. The command to train is:<br>
```bash
python fed\_train.py --disable\_pretrain --total\_size 30 --edge\_fed\_interval 1 --cloud\_fed\_interval 1
```

4. Limited data & Enable pretrain:
---
In this case, the data for FL is limited and it enables the pretraining stage to train a model. Owing to transferring some training data to cloud server, so less
epochs is for the training. It still does the edge federated learning and cloud federated learning at every epoch in this case. The command to train is:<br>
```bash
python fed\_train.py --total\_size 30 --edge\_fed\_interval 1 --cloud\_fed\_interval 1 --pretrain\_epochs 20 --pretrain\_batch\_cnt 50
```

5. Limited data & Enable pretrain & Optimized federated interval(both edge and cloud):
---
Compared with the previous case, in this case it just changes the edge federated learning interval and cloud federated learning interval. The command to train is:<br>
```bash
python fed\_train.py --total\_size 30 --edge\_fed\_interval 2 --cloud\_fed\_interval 2 --pretrain\_epochs 20 --pretrain\_batch\_cnt 50
```

6. Limited data & Enable pretrain with more batches of training data & Optimized federated interval(both edge and cloud):
---
Compared with the previous case, this case it just upload more training data to cloud server to do pretrain. The command to train is:<br>
```bash
python fed\_train.py --total\_size 30 --edge\_fed\_interval 2 --cloud\_fed\_interval 2 --pretrain\_epochs 20 --pretrain\_batch\_cnt 60
```

PS: It is noted that for all the limited data cases, the training process stops automatically when the data is used up!
