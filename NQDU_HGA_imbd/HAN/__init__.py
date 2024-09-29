import copy
import time
import math
import torch
import numpy as np
# from dataset import GraphSourceTargetDataset
from HAN.revmodel import *
from HAN.utils import *
from HAN.evaluate import *
class GraphSourceTargetDataset(object):
    def __init__(self, args):
        self.args = args
        # set_seed(args.seed, args.device)
        self.adjsS, self.id_nameS, self.featuresS = load_data_semisupervisedS(args, args.nodeS, args.linkS, args.configS,
                                                                             list(map(lambda x: int(x),
                                                                                      args.metaS.split(','))))
        self.train_poolS, self.train_labelS_np, self.nlabelS, self.multiS = load_label(args.labelS, self.id_nameS)
        self.adjsT, self.id_nameT, self.featuresT = load_data_semisupervisedT(args, args.nodeT, args.linkT, args.configT,
                                                                             list(map(lambda x: int(x),
                                                                                      args.metaT.split(','))))
        self.train_poolT, self.train_labelT_np, self.nlabelT, self.multiT = load_label(args.labelT, self.id_nameT)
        # print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 'finish loading', flush=True)
        self.nhead = list(map(lambda x: int(x), args.nhead.split(',')))
        self.nnodeS, self.nchannel, self.nlayer = len(self.id_nameS), len(self.adjsS), len(self.nhead)
        self.nnodeT, self.nchannel, self.nlayer = len(self.id_nameT), len(self.adjsT), len(self.nhead)

        if args.attributed == 'True': self.nfeat = self.featuresS.shape[1]

        self.embeddingsS = torch.from_numpy(self.featuresS).to(
            args.device) if args.attributed == 'True' else torch.from_numpy(
            np.random.randn(self.nnodeS, args.size).astype(np.float32)).to(args.device)
        self.embeddingsT = torch.from_numpy(self.featuresT).to(
            args.device) if args.attributed == 'True' else torch.from_numpy(
            np.random.randn(self.nnodeT, args.size).astype(np.float32)).to(args.device)

        self.train_labelS = torch.from_numpy(self.train_labelS_np.astype(np.float32)).to(args.device)
        # print('train_labelS',train_labelS.shape)
        self.train_labelT = torch.from_numpy(self.train_labelT_np.astype(np.float32)).to(args.device)


class HAN_Trainer(object):

    def __init__(self, args, dataset:GraphSourceTargetDataset):
        self.args = args
        self.source_target_dataset = dataset
        self.model = RevGrad(self.source_target_dataset.nchannel, self.source_target_dataset.nfeat, args.size, self.source_target_dataset.nlabelS, self.source_target_dataset.nlayer, self.source_target_dataset.nhead, args.neigh_por, args.dropout, args.alpha, args.device)
        self.model.to(args.device)
        

    def fit(self, is_train_dataset, batch_size, epochs, sample_weights=None, sample_indices=None,Flag=None):
        if is_train_dataset: # for source training
            nnodeS = self.source_target_dataset.nnodeS
            nnodeT = self.source_target_dataset.nnodeT
            train_poolS = self.source_target_dataset.train_poolS
            train_poolT = self.source_target_dataset.train_poolT
            train_labelS = self.source_target_dataset.train_labelS
            embeddingsS = self.source_target_dataset.embeddingsS
            embeddingsT = self.source_target_dataset.embeddingsT
            adjsS = self.source_target_dataset.adjsS
            adjsT = self.source_target_dataset.adjsT
        else: # for target training
            nnodeS = self.source_target_dataset.nnodeT
            nnodeT = self.source_target_dataset.nnodeT
            train_poolS = self.source_target_dataset.train_poolT
            train_poolT = self.source_target_dataset.train_poolT
            train_labelS = self.source_target_dataset.train_labelT
            embeddingsS = self.source_target_dataset.embeddingsT
            embeddingsT = self.source_target_dataset.embeddingsT
            adjsS = self.source_target_dataset.adjsT
            adjsT = self.source_target_dataset.adjsT


        for epoch in range(epochs):
                self.model.train()
                LEARNING_RATE = self.args.lr / math.pow((1 + 10 * epoch / epochs), 0.75)
                # print("------------------------start this epoch--------------------------------")
                print("learning rate: ", LEARNING_RATE)

                optimizer = torch.optim.Adam([{'params': self.model.sharedNet.parameters()},
                {'params': self.model.linear_block.parameters(), 'lr': LEARNING_RATE},
                {'params': self.model.HeteroAttLayerT.parameters(), 'lr': LEARNING_RATE},
                {'params': self.model.cls_fc1.parameters(), 'lr': LEARNING_RATE},
                {'params': self.model.cls_fc2.parameters(), 'lr': LEARNING_RATE}], lr=LEARNING_RATE/10, weight_decay=self.args.weight_decay)

                batch_size = int(batch_size)
                if sample_indices is None:
                    num_iter = math.ceil(nnodeS/batch_size)
                else:
                    num_iter = math.ceil(len(sample_indices)/batch_size)
                # print('num_iter',num_iter)
                for i in range(1, num_iter):
                    curr_indexS = np.sort(np.random.choice(np.arange(len(train_poolS)), self.args.inner_batch_size, replace=False))
                    # if sample_indices is not None:
                    #     curr_indexS = sample_indices[(i-1)*batch_size:i*batch_size]
                    # curr_batchS = train_poolS[curr_indexS]
                    # if sample_weights is not None:
                    #     if sample_indices is not None:
                    #         sample_weight = sample_weights[(i-1)*batch_size:i*batch_size]
                    #     else:
                    #         sample_weight = sample_weights[curr_indexS]
                    # else:
                    #     sample_weight = None

                    if sample_indices is not None:
                        tmp_indexS = np.sort(
                            np.random.choice(np.arange(len(sample_indices)), self.args.inner_batch_size, replace=False))
                        curr_indexS = sample_indices[tmp_indexS]
                    curr_batchS = train_poolS[curr_indexS]
                    if sample_weights is not None:
                        if sample_indices is not None:
                            tmp_indexS = np.sort(
                                np.random.choice(np.arange(len(sample_indices)), self.args.inner_batch_size,
                                                 replace=False))
                            curr_indexS = sample_indices[tmp_indexS]
                            sample_weight = sample_weights[tmp_indexS]
                        else:
                            sample_weight = sample_weights[curr_indexS]
                    else:
                        sample_weight = None

                    curr_indexT = np.sort(np.random.choice(np.arange(len(train_poolT)), self.args.inner_batch_size, replace=False))
                    curr_batchT = train_poolT[curr_indexT]

                    optimizer.zero_grad()
                    cls_lossS,cls_lossT,  mmd_loss, l1_loss, _ ,cls_lossS_MP,homo_outT,new_hsT= self.model(embeddingsS, adjsS, curr_batchS, embeddingsT, adjsT, curr_batchT, train_labelS[curr_indexS],Training=True,mark=0, sample_weight=sample_weight)

                    gamma = 2 / (1 + math.exp(-10 * (epoch + 1) / self.args.epochs)) - 1
                    # gamma=0.1
                    loss = cls_lossS + gamma * (mmd_loss + l1_loss)+cls_lossT+cls_lossS_MP
                    loss.backward()
                    optimizer.step()

                    optimizer.zero_grad()
                    cls_lossS,cls_lossT, mmd_loss, l1_loss, _,cls_lossS_MP,homo_outT,new_hsT= self.model(embeddingsS, adjsS, curr_batchS, embeddingsT, adjsT, curr_batchT, train_labelS[curr_indexS],Training=True,mark=1, sample_weight=sample_weight)

                    gamma = 2 / (1 + math.exp(-10 * (epoch + 1) / self.args.epochs)) - 1
                    # gamma=0.1
                    loss = cls_lossS + gamma * (mmd_loss + l1_loss)+cls_lossT+cls_lossS_MP
                    loss.backward()
                    optimizer.step()
                    print('Train Epoch: {}\ti: {}\tLoss: {:.6f}'.format(epoch, i, loss.item()))
                    if Flag == 'final':
                        with open('DVE final model loss result.txt', 'a', encoding='utf-8') as f:
                            f.write(str(epoch) + '\t' + str(i) + '\t' + str(loss.item()) + '\n')
                        f.close()


                # print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + f'finish epoch {epoch}', flush=True)

                    
                # print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 'output embedding', flush=True)
                # print("--------------------------end this epoch------------------------------")

    def predict(self, is_train_dataset, batch_size, proba=False):
        if is_train_dataset: # for source training
            nnodeT = self.source_target_dataset.nnodeS
            embeddingsS = self.source_target_dataset.embeddingsS
            adjsS = self.source_target_dataset.adjsS
            # embeddingsT = self.source_target_dataset.embeddingsT
            # adjsT = self.source_target_dataset.adjsT
            embeddingsT = self.source_target_dataset.embeddingsS
            adjsT = self.source_target_dataset.adjsS
        else:
            nnodeT = self.source_target_dataset.nnodeT
            embeddingsS = self.source_target_dataset.embeddingsT
            adjsS = self.source_target_dataset.adjsT
            embeddingsT = self.source_target_dataset.embeddingsT
            adjsT = self.source_target_dataset.adjsT

        rounds = math.ceil(nnodeT/batch_size)
        outputs = np.zeros((nnodeT, 3)).astype(np.float32)
        homo_outT_1=np.zeros((nnodeT, 512)).astype(np.float32)
        homo_outT_2=np.zeros((nnodeT, 512)).astype(np.float32)
        new_hsT_1=np.zeros((nnodeT, 64)).astype(np.float32)
        new_hsT_2 = np.zeros((nnodeT, 64)).astype(np.float32)
        with torch.no_grad():
            for index, i in enumerate(range(rounds)):
                seed_nodes = np.arange(i*batch_size, min((i+1)*batch_size, nnodeT))

                _,cls_loss, mmd_loss, l1_loss, c_pred, _,homo_outT_a,new_hsT_a = self.model(embeddingsS, adjsS, seed_nodes, embeddingsT, adjsT, seed_nodes, None, Training=True,mark=-1)
                # train_labelS[curr_indexS]这个参数没用上
                target_probs = F.softmax(c_pred, dim=-1)
                outputs[seed_nodes] = target_probs.detach().cpu().numpy()
                homo_outT_a=homo_outT_a.detach().cpu().numpy()
                homo_outT_1[seed_nodes]=homo_outT_a[0]
                homo_outT_2[seed_nodes] = homo_outT_a[1]
                new_hsT_a=new_hsT_a.detach().cpu().numpy()
                new_hsT_1[seed_nodes] = new_hsT_a[0]
                new_hsT_2[seed_nodes] = new_hsT_a[1]
                # print('here')
                # print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + f'finish output batch {index} -> {rounds}', flush=True)

        # print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ",
                            # time.localtime()) + f'finish predict', flush=True)
        homo_outT=np.vstack((homo_outT_1,homo_outT_2))
        new_hsT = np.vstack((new_hsT_1, new_hsT_2))
        if proba:
            return outputs,homo_outT,new_hsT
        else:
            return np.argmax(outputs, axis=1),homo_outT,new_hsT


    def copy(self):
        # new_model = RevGrad(self.nchannel, self.nfeat, self.args.size, self.nlabelS, self.nlayer, self.nhead, self.args.neigh_por, self.args.dropout, self.args.alpha, self.args.device)
        new_model = self.model
        copy_model = copy.deepcopy(self)
        copy_model.model = new_model
        return copy_model


    def save_model(self, save_path):
        torch.save(self.model.state_dict(),save_path)

    def load_model(self, save_path):
        self.model.load_state_dict(torch.load(save_path))