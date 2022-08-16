import copy
import torch
import torch.nn as nn
import numpy as np
import threading
import torch.multiprocessing as mp
import os
from tqdm import tqdm

from .fed_client import Client
from .fed_optim_variables import *

import sys
sys.path.append("../")

from fusion import ActionPredModel
from config import *
from utils.one_hot import label2onehot

class EdgeServer():
    def __init__(self, server_id, config, aug_seq, edges_dict, training_config, tensorboard):
        self.id = server_id
        self.clients_num = len(config)
        self.clients_dict = {}
        self.clients_log = {}
        self.avgModel = None
        self.edges_dict = edges_dict
        self.epochs = training_config['epochs']

        self.clients = []
        for i in range(self.clients_num):
            cid = 'vehicle'+str(i)
            print('Client %s.%s is initialized!' % (self.id, cid))
            self.clients.append(Client(self.id, cid, config[cid], aug_seq, self.clients_dict,
                                       self.clients_log, training_config, tensorboard))

    def run(self):
        for i in range(self.clients_num):
            self.clients[i].train()

    def FedAvg(self):
        w_avg = copy.deepcopy(self.clients_dict['vehicle0']['model_dict'])
        for k in w_avg.keys():
            for i in range(1, len(self.clients_dict)):
                w_avg[k] += self.clients_dict['vehicle' + str(i)]['model_dict'][k]
            w_avg[k] = torch.div(w_avg[k], len(self.clients_dict))

        self.avgModel = w_avg
        self.edges_dict[self.id] = self.avgModel

    def SinkModelToClients(self):
        for client in self.clients:
            client.updated_model = self.avgModel

    def PreparePretrainData(self):
        data = []
        for client in self.clients:
            data += client.PreparePretrainData()
        return data

class CloudServer():
    def __init__(self, aug_seq, training_config, tensorboard):
        from .fed_config import config
        self.edges_num = len(config)
        self.edges_dict = {}
        self.pretrain_data = None
        self.pretrain_model = ActionPredModel().to(device)
        self.avgModel = self.pretrain_model.state_dict()
        self.pretrain_config = training_config
        self.tb = tensorboard
        self.epochs = training_config['epochs']

        self.edges = []
        for i in range(self.edges_num):
            eid = 'edge'+str(i)
            self.edges.append(EdgeServer(eid, config[eid], aug_seq, self.edges_dict,
                                         training_config, tensorboard))

    def train(self):
        edge_fed_cnt = 0
        for j in range(self.epochs // edge_fed_interval):
            for edge in self.edges:
                edge.run()

            for edge in sefl.edges:
                edge.FedAvg()
                edge.SinkModelToClients()
                print("[Edge Federated Stage] [%s Federated!]" % edge.id)

            edge_fed_cnt += 1
            if edge_fed_cnt == cloud_fed_interval:
                edge_fed_cnt = 0
                self.FedAvg()
                self.SinkModelToEdges()
                for edge in self.edges:
                    edge.SinkModelToClients()
                print("[Cloud Federated Stage] [CloudServer Federated!]")
                save_path = ACTION_MODEL_PATH + "FL_%d_action.pth" % j
                torch.save(self.avgModel, save_path)

    def FedAvg(self):
        for edge in self.edges:
            edge.FedAvg()

        w_avg = copy.deepcopy(self.edges_dict['edge0'])
        for k in w_avg.keys():
            for i in range(1, len(self.edges_dict)):
                w_avg[k] += self.edges_dict['edge' + str(i)][k]
            w_avg[k] = torch.div(w_avg[k], len(self.edges_dict))

        self.avgModel = w_avg

    def SinkModelToEdges(self):
        for edge in self.edges:
            edge.avgModel = self.avgModel

    def QueryTrainingLog(self):
        log = {}
        for edge in self.edges:
            log[edge.id] = edges.clients_log
        return log

    def CollectPretrainData(self):
        data = []
        for edge in self.edges:
            data += edge.PreparePretrainData()

        self.pretrain_data = data

    def Pretrain(self):
        softmax = nn.Softmax(dim=1)

        mse_loss = nn.MSELoss().to(device)
        mce_loss = nn.CrossEntropyLoss().to(device)

        parameters = self.pretrain_model.parameters()
        optim = torch.optim.Adam(parameters, lr=self.pretrain_config['lr'], betas=self.pretrain_config['betas'], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.1)

        batch_cnt = len(self.pretrain_data)
        data_len = batch_cnt * self.pretrain_config['batch_size']
        for epoch in range(pretrain_epochs):
            self.pretrain_model.train()

            avg_loss1_sum = 0.0
            avg_loss2_sum = 0.0
            right_cnt = 0
            for data in tqdm(self.pretrain_data):
                params1, params2 = data['params1'], data['params2']
                st_action, tb_action = data['st_action'], data['tb_action']
                _, imgs_f, imgs_l, imgs_r = params1[0], params1[1], params1[2], params1[3]
                _, imgs_b = params2[0], params2[1]

                optim.zero_grad()

                imgs_f = imgs_f.to(device)
                imgs_l = imgs_l.to(device)
                imgs_r = imgs_r.to(device)
                imgs_b = imgs_b.to(device)

                out_thro_brake, out_steer = self.pretrain_model(imgs_f, imgs_l, imgs_r, imgs_b)

                thro_brake = tb_action.reshape([self.pretrain_config['batch_size'], -1])
                avg_loss1 = mse_loss(out_thro_brake, Tensor(thro_brake))

                steer = st_action.reshape([self.pretrain_config['batch_size'], -1])
                label = steer.squeeze()
                steer = Tensor(label2onehot(steer, 7))
                avg_loss2 = mce_loss(out_steer, steer)

                avg_loss1.backward()
                avg_loss2.backward()

                optim.step()

                avg_loss1_sum += avg_loss1
                avg_loss2_sum += avg_loss2

                out_prob = softmax(out_steer)

                out_prob = np.argmax(out_prob.detach().cpu().numpy(), axis=1)

                pred_res = out_prob==np.array(label)
                right_cnt += sum(pred_res)

            scheduler.step()

            loss1 = avg_loss1_sum / batch_cnt
            loss2 = avg_loss2_sum / batch_cnt

            steer_acc = right_cnt / data_len

            log_info = "[Pretraining Stage] [Epoch: %d/%d] [ThrottleBrake Loss: %f, Steer Loss: %f, Steer Accuracy: %f]" % (epoch, pretrain_epochs, loss1.item(), loss2.item(), steer_acc)
            print(log_info)

            self.tb.add_scalar('PretrainStage.ThrottheBrakeLoss', loss1, epoch)
            self.tb.add_scalar('PretrainStage.SteerLoss', loss2, epoch)
            self.tb.add_scalar('PretrainStage.SteerAccuracy', steer_acc, epoch)

            if epoch == pretrain_epochs - 1:
                self.avgModel = self.pretrain_model.state_dict()
                del self.pretrain_model

    def CalcClientsCnt(self):
        cnt = 0
        for edge in self.edges:
            cnt += edge.clients_num

        return cnt
