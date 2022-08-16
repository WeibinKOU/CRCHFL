import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm

from .fed_optim_variables import *

import sys
sys.path.append("../")

from fusion import ActionPredModel
from dataloader import ImgData, ActionData, MultiImgData
from config import *
from utils.one_hot import label2onehot

class Client():
    def __init__(self, server_id, client_id, config, aug_seq, clients_dict, clients_log, training_config, tensorboard):
        self.sid = server_id
        self.cid = client_id
        self.config = config
        self.tb = tensorboard
        self.epochs = training_config['epochs']
        self.batch_size = training_config['batch_size']
        self.clients_dict = clients_dict
        self.clients_log = clients_log
        self.model = ActionPredModel().to(device)
        self.updated_model = None
        self.epoch_cnt = 0

        self.steer_data = MultiImgData(config['steer_data'], aug_seq, data_transform)
        self.steer_action = ActionData(config['steer_action'])
        self.thro_brake_data = ImgData(config['thro_brake_data'], aug_seq, data_transform)
        self.thro_brake_action = ActionData(config['thro_brake_action'])

        self.steer_dl = DataLoader(self.steer_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True)
        self.thro_brake_dl = DataLoader(self.thro_brake_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True)
        self.data_len = len(self.steer_dl) * self.batch_size

        self.mce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        self.softmax = nn.Softmax(dim=1)

        self.parameters = self.model.parameters()
        self.optim = torch.optim.Adam(self.parameters,
                                      lr=training_config['lr'],
                                      betas=training_config['betas'],
                                      weight_decay=training_config['weight_decay'])

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=20, gamma=0.1)

        self.batch_cnt = min(len(self.steer_dl), len(self.thro_brake_dl))

        self.dict = {
                'model_dict' : None
            }
        self.log = {
                'epochs' : 0,
                'steer_loss' : 0.0,
                'steer_acc' : 0.0,
                'thro_brake_loss' : 0.0,
                'log_info' : None
            }

    def PreparePretrainData(self):
        steer_dl = copy.deepcopy(self.steer_dl)
        thro_brake_dl = copy.deepcopy(self.thro_brake_dl)
        steer_action = copy.deepcopy(self.steer_action)
        thro_brake_action = copy.deepcopy(self.thro_brake_action)

        data_list = []
        for i in range(pretrain_batch_cnt):
            params1, params2 = next(iter(steer_dl)), next(iter(thro_brake_dl))

            st_action = steer_action.getitems(params1[0])[:, 1]
            tb_action = thro_brake_action.getitems(params2[0])[:, 0:3:2]

            data = {}
            data['params1'] = params1
            data['params2'] = params2
            data['st_action'] = st_action
            data['tb_action'] = tb_action

            data_list.append(data)

        del steer_dl
        del thro_brake_dl
        del steer_action
        del thro_brake_action

        return data_list

    def train(self):
        for epoch in range(edge_fed_interval):
            if self.updated_model is not None:
                self.model.load_state_dict(self.updated_model)
                self.updated_model = None

            self.model.train()

            avg_loss1_sum = 0.0
            avg_loss2_sum = 0.0
            right_cnt = 0
            for iteration in tqdm(range(self.batch_cnt)):
                params1, params2 = next(iter(self.steer_dl)), next(iter(self.thro_brake_dl))
                names1, imgs_f, imgs_l, imgs_r = params1[0], params1[1], params1[2], params1[3]
                names2, imgs_b = params2[0], params2[1]

                st_action = self.steer_action.getitems(names1)[:, 1]
                tb_action = self.thro_brake_action.getitems(names2)[:, 0:3:2]

                self.optim.zero_grad()

                imgs_f = imgs_f.to(device)
                imgs_l = imgs_l.to(device)
                imgs_r = imgs_r.to(device)
                imgs_b = imgs_b.to(device)

                out_thro_brake, out_steer = self.model(imgs_f, imgs_l, imgs_r, imgs_b)

                thro_brake = tb_action.reshape([self.batch_size, -1])
                avg_loss1 = self.mse_loss(out_thro_brake, Tensor(thro_brake))

                steer = st_action.reshape([self.batch_size, -1])
                label = steer.squeeze()
                steer = Tensor(label2onehot(steer, 7))
                avg_loss2 = self.mce_loss(out_steer, steer)

                avg_loss1.backward()
                avg_loss2.backward()

                self.optim.step()

                avg_loss1_sum += avg_loss1
                avg_loss2_sum += avg_loss2

                out_prob = self.softmax(out_steer)

                out_prob = np.argmax(out_prob.detach().cpu().numpy(), axis=1)

                pred_res = out_prob==np.array(label)
                right_cnt += sum(pred_res)

            self.lr_scheduler.step()

            self.dict['model_dict'] = self.model.state_dict()

            self.log['thro_brake_loss'] = avg_loss1_sum / self.batch_cnt
            self.log['steer_loss'] = avg_loss2_sum / self.batch_cnt

            self.log['steer_acc'] = right_cnt / self.data_len
            self.log['epochs'] = epoch

            self.log['log_info'] = "[Client ID: %s.%s, Epoch: %d/%d] [ThrottleBrake Loss: %f, Steer Loss: %f, Steer Accuracy: %f]" % (self.sid,
                              self.cid, self.epoch_cnt, self.epochs,
                              self.log['thro_brake_loss'].item(),
                              self.log['steer_loss'].item(),
                              self.log['steer_acc'])

            self.clients_dict[self.cid] = self.dict
            self.clients_log[self.cid] = self.log

            self.tb.add_scalar(self.sid + '.' + self.cid + '.ThrottleBrakeLoss', self.log['thro_brake_loss'], self.epoch_cnt)
            self.tb.add_scalar(self.sid + '.' + self.cid + '.SteerLoss', self.log['steer_loss'], self.epoch_cnt)
            self.tb.add_scalar(self.sid + '.' + self.cid + '.SteerAccuracy', self.log['steer_acc'], self.epoch_cnt)
            print(self.log['log_info'])

            self.epoch_cnt += 1
