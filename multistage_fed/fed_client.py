import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm

import sys
sys.path.append("../")

from fusion import ActionPredModel
from dataloader import ImgData, ActionData, MultiImgData
from config import *
from utils.one_hot import label2onehot

class Client():
    def __init__(self, server_id, client_id, config, test_data, aug_seq, clients_dict, clients_log, training_config, tensorboard, scheduler):
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
        self.eval_data = test_data
        self.fed_cnt = 0
        self.softmax = nn.Softmax(dim=1)
        self.mse_loss = nn.MSELoss().to(device)
        self.mce_loss = nn.CrossEntropyLoss().to(device)
        self.scheduler = scheduler

        self.dataset = MultiImgData(config['dataset'], aug_seq, data_transform)
        self.action = ActionData(config['action'])

        self.dataloader = DataLoader(self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True)
        self.data_len = len(self.dataloader) * self.batch_size

        self.parameters = self.model.parameters()
        self.optim = torch.optim.Adam(self.parameters,
                                      lr=training_config['lr'],
                                      betas=training_config['betas'],
                                      weight_decay=training_config['weight_decay'])

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=20, gamma=0.1)

        self.batch_cnt = len(self.dataloader)

        self.log = {
                'epochs' : 0,
                'steer_loss' : 0.0,
                'steer_acc' : 0.0,
                'thro_brake_loss' : 0.0,
                'log_info' : None
            }

    def PreparePretrainData(self):
        data = {}
        data['loader'] = self.dataloader
        data['action'] = self.action

        ret = self.scheduler.transfer_entries(self.scheduler.pretrain_batch_cnt * self.batch_size)
        if not ret:
            print("Initialized data size is not enough to transfer pretaining data, so exit!")
            sys.exit()

        return data

    def PrepareTrainData(self):
        data = {}
        data['loader'] = self.dataloader
        data['action'] = self.action
        return data

    def Evaluate(self):
        self.model.eval()

        dataloader, action, data_len, batch_cnt = self.eval_data

        right_cnt = 0
        avg_loss1_sum = 0.0
        avg_loss2_sum = 0.0

        loss = 0.0
        steer_acc = 0.0

        with torch.no_grad():
            for names, imgs_f, imgs_l, imgs_r, imgs_b in tqdm(dataloader):
                action_batch = action.getitems(names)
                st_action, tb_action = action_batch[:, 1], action_batch[:, 0:3:2]

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

                avg_loss1_sum += avg_loss1
                avg_loss2_sum += avg_loss2

                out_prob = self.softmax(out_steer)

                out_prob = np.argmax(out_prob.detach().cpu().numpy(), axis=1)

                pred_res = out_prob==np.array(label)
                right_cnt += sum(pred_res)

            loss1 = avg_loss1_sum / batch_cnt
            loss2 = avg_loss2_sum / batch_cnt
            loss = loss1.item() + loss2.item()

            steer_acc = right_cnt / data_len

        return loss, steer_acc

    def train(self):
        for epoch in range(self.scheduler.edge_fed_interval):
            if self.updated_model is not None:
                self.model.load_state_dict(self.updated_model)
                self.updated_model = None

            self.model.train()

            avg_loss1_sum = 0.0
            avg_loss2_sum = 0.0
            right_cnt = 0
            for names, imgs_f, imgs_l, imgs_r, imgs_b in tqdm(self.dataloader):
                action_batch = self.action.getitems(names)
                st_action, tb_action = action_batch[:, 1], action_batch[:, 0:3:2]

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

            self.log['thro_brake_loss'] = avg_loss1_sum / self.batch_cnt
            self.log['steer_loss'] = avg_loss2_sum / self.batch_cnt

            self.log['steer_acc'] = right_cnt / self.data_len
            self.log['epochs'] = epoch

            cid = "%s.%s" % (self.sid, self.cid)
            self.log['log_info'] = "[Epoch: %d/%d] [%s.Train.Loss: %f, %s.Train.Accuracy: %f]" % (self.epoch_cnt, self.epochs, cid, self.log['thro_brake_loss'].item() + self.log['steer_loss'].item(), cid, self.log['steer_acc'])
            print(self.log['log_info'])

            self.clients_log[self.cid] = self.log

            self.tb.add_scalar(cid + '.Train.Loss',
                               self.log['thro_brake_loss'] + self.log['steer_loss'], self.epoch_cnt)
            self.tb.add_scalar(cid + '.Train.Accuracy', self.log['steer_acc'], self.epoch_cnt)

            eval_loss, eval_acc = self.Evaluate()
            self.tb.add_scalar(cid + '.Eval.Loss', eval_loss, self.epoch_cnt)
            self.tb.add_scalar(cid + '.Eval.Loss', eval_acc, self.epoch_cnt)
            log_info = "[Epoch: %d/%d] [%s.Eval.Loss: %f, %s.Eval.Accuracy: %f]" % (self.epoch_cnt, self.epochs, cid, eval_loss, cid, eval_acc)
            print(log_info)

            self.epoch_cnt += 1

            if epoch == self.scheduler.edge_fed_interval - 1:
                self.clients_dict[self.cid] = self.model.state_dict()
                ret = self.scheduler.transfer_model()
                if not ret:
                    print("Initialized data size has been used up, so exit!")
                    sys.exit()
