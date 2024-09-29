import copy
import time
import math
import torch
import numpy as np
# from dataset import GraphSourceTargetDataset
from ACDNE.ACDNE_model import *
# from HAN.utils import *
# from HAN.evaluate import *
from ACDNE.utils import *
from ACDNE.flip_gradient import GradReverse
from torch.nn import functional as F
from scipy.sparse import lil_matrix
class GraphSourceTargetDataset(object):
    def __init__(self, args):
        self.args = args
        set_seed(args.seed, args.device)
        self.input_data = load_source_target_data(args)
        self.adj_s = self.input_data['adj_S']
        self.adj_t = self.input_data['adj_T']

        self.ppmi_s = self.input_data['PPMI_S']
        self.ppmi_t = self.input_data['PPMI_T']
        # self.ppmi_s = torch.from_numpy(self.ppmi_s).to(args.device)
        # self.ppmi_t = torch.from_numpy(self.ppmi_t).to(args.device)

        self.x_s = self.input_data['attrb_S']
        self.x_t = self.input_data['attrb_T']
        # self.x_s = torch.from_numpy(lil_matrix.toarray(self.x_s)).to(args.device)
        # self.x_t = torch.from_numpy(lil_matrix.toarray(self.x_t)).to(args.device)

        self.x_n_s = self.input_data['attrb_nei_S']
        self.x_n_t = self.input_data['attrb_nei_T']
        # self.x_n_s = torch.from_numpy(self.x_n_s).to(args.device)
        # self.x_n_t = torch.from_numpy(self.x_n_t).to(args.device)

        self.y_s = self.input_data['label_S']
        self.y_t = self.input_data['label_T']
        # self.y_s = torch.from_numpy(self.y_s).to(args.device)
        # self.y_t = torch.from_numpy(self.y_t).to(args.device)

        self.y_t_o = np.zeros(np.shape(self.y_t))  # observable label matrix of target network, all zeros
        # self.y_t_o = torch.from_numpy(self.y_t_o).to(args.device)

        self.x_s_new = lil_matrix(np.concatenate((lil_matrix.toarray(self.x_s), self.x_n_s), axis=1))
        self.x_t_new = lil_matrix(np.concatenate((lil_matrix.toarray(self.x_t), self.x_n_t), axis=1))

        self.n_input = self.x_s.shape[1]
        self.num_class = self.y_s.shape[1]
        self.num_nodes_s = self.x_s.shape[0]
        self.num_nodes_t = self.x_t.shape[0]
        self.whole_xs_xt_stt = torch.FloatTensor(vstack([self.x_s, self.x_t]).toarray())
        self.whole_xs_xt_stt_nei = torch.FloatTensor(vstack([self.x_n_s, self.x_n_t]).toarray())
        self.whole_xs_xt_stt = self.whole_xs_xt_stt.to(args.device)
        self.whole_xs_xt_stt_nei = self.whole_xs_xt_stt_nei.to(args.device)



class ACDNE_Trainer(object):

    def __init__(self, args, dataset:GraphSourceTargetDataset):
        self.args = args
        self.source_target_dataset = dataset
        self.model =ACDNE(self.source_target_dataset.n_input, args.n_hidden, args.n_emb, self.source_target_dataset.num_class, args.clf_type, args.inner_batch_size, args.dropout)
        self.model.to(args.device)

        # self.clf_loss_f = nn.BCEWithLogitsLoss(reduction='none') if self.model.node_classifier.clf_type == 'multi-label' \
        #     else nn.CrossEntropyLoss()
        # self.domain_loss_f = nn.CrossEntropyLoss()
        self.f1_t = None

    def fit(self, is_train_dataset, batch_size, epochs, sample_weights=None, sample_indices=None,Flag='None'):
        if is_train_dataset:  # for source training
            x_s_new = self.source_target_dataset.x_s_new
            y_s = self.source_target_dataset.y_s
            x_t_new = self.source_target_dataset.x_t_new
            y_t_o = self.source_target_dataset.y_t_o
            num_nodes_s = self.source_target_dataset.num_nodes_s
            num_nodes_t = self.source_target_dataset.num_nodes_t
            n_input = self.source_target_dataset.x_s.shape[1]
            ppmi_s = self.source_target_dataset.ppmi_s
            ppmi_t = self.source_target_dataset.ppmi_t


            ##
            y_t = self.source_target_dataset.y_t
            x_s = self.source_target_dataset.x_s
            x_t = self.source_target_dataset.x_t
            x_n_s = self.source_target_dataset.x_n_s
            x_n_t = self.source_target_dataset.x_n_t
            whole_xs_xt_stt = torch.FloatTensor(vstack([x_s, x_t]).toarray())

            a = lil_matrix(x_n_s)
            b = lil_matrix(x_n_t)
            c = vstack([a, b])
            whole_xs_xt_stt_nei = torch.FloatTensor(c.toarray())

            whole_xs_xt_stt = whole_xs_xt_stt.to(self.args.device)
            whole_xs_xt_stt_nei = whole_xs_xt_stt_nei.to(self.args.device)
            ##

        else:
            x_s_new = self.source_target_dataset.x_t_new
            y_s = self.source_target_dataset.y_t
            x_t_new = self.source_target_dataset.x_t_new
            y_t_o = self.source_target_dataset.y_t_o
            num_nodes_s = self.source_target_dataset.num_nodes_t
            num_nodes_t = self.source_target_dataset.num_nodes_t
            n_input = self.source_target_dataset.x_t.shape[1]
            ppmi_s = self.source_target_dataset.ppmi_t
            ppmi_t = self.source_target_dataset.ppmi_t

            ##
            y_t = self.source_target_dataset.y_t
            x_s = self.source_target_dataset.x_t
            x_t = self.source_target_dataset.x_t
            x_n_s = self.source_target_dataset.x_n_t
            x_n_t = self.source_target_dataset.x_n_t
            whole_xs_xt_stt = torch.FloatTensor(vstack([x_s, x_t]).toarray())

            a=lil_matrix(x_n_s)
            b=lil_matrix(x_n_t)
            c=vstack([a,b])
            whole_xs_xt_stt_nei = torch.FloatTensor(c.toarray())
            whole_xs_xt_stt = whole_xs_xt_stt.to(self.args.device)
            whole_xs_xt_stt_nei = whole_xs_xt_stt_nei.to(self.args.device)
            ##

        if sample_indices is not None:
            x_s_new = lil_matrix(lil_matrix.toarray(x_s_new)[sample_indices])
            y_s = y_s[sample_indices]
            num_nodes_s = y_s.shape[0]
            ppmi_s = ppmi_s[sample_indices]

        max_micro = 0
        max_macro = 0
        max_epoch = 0
        micro=[]
        macro=[]
        for cEpoch in range(epochs):

            s_batches = batch_generator([x_s_new, y_s], int(batch_size / 2), shuffle=True)
            t_batches = batch_generator([x_t_new, y_t_o], int(batch_size / 2), shuffle=True)
            num_batch = round(max(num_nodes_s / (batch_size / 2), num_nodes_t / (batch_size / 2)))
            # Adaptation param and learning rate schedule as described in the DANN paper
            p = float(cEpoch) / epochs
            lr = self.args.lr_ini / (1. + 10 * p) ** 0.75
            grl_lambda = 2. / (1. + np.exp(-10. * p)) - 1  # gradually change from 0 to 1
            GradReverse.rate = grl_lambda
            # optimizer = torch.optim.SGD(self.model.parameters(), lr, 0.9, weight_decay=self.args.l2_w / 2)
            # optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay=l2_w/2)
            optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=self.args.l2_w/2)
            for cBatch in range(num_batch):
                # ## each batch, half nodes from source network, and half nodes from target network
                xs_ys_batch, shuffle_index_s = next(s_batches)
                if sample_weights is not None:
                    sample_weight = sample_weights[shuffle_index_s]
                    sample_weight = torch.from_numpy(sample_weight).to('cuda')
                else:
                    sample_weight = None

                xs_batch = xs_ys_batch[0]
                ys_batch = xs_ys_batch[1]
                xt_yt_batch, shuffle_index_t = next(t_batches)
                xt_batch = xt_yt_batch[0]
                yt_batch = xt_yt_batch[1]
                x_batch = vstack([xs_batch, xt_batch])
                batch_csr = x_batch.tocsr()
                xb = torch.FloatTensor(batch_csr[:, 0:n_input].toarray())
                xb_nei = torch.FloatTensor(batch_csr[:, -n_input:].toarray())
                xb = xb.to(self.args.device)
                xb_nei = xb_nei.to(self.args.device)
                yb = np.vstack([ys_batch, yt_batch])

                mask_l = np.sum(yb, axis=1) > 0
                # 1 if the node is with observed label, 0 if the node is without label
                domain_label = np.vstack([np.tile([1., 0.], [batch_size // 2, 1]), np.tile([0., 1.], [
                    batch_size // 2, 1])])  # [1,0] for source, [0,1] for target
                # #topological proximity matrix between nodes in each mini-batch
                a_s, a_t = batch_ppmi(batch_size, shuffle_index_s, shuffle_index_t, ppmi_s, ppmi_t)
                # a_s = torch.from_numpy(a_s).to(self.args.device)
                # a_s = torch.tensor(a_s, dtype=torch.float32)
                # a_t = torch.from_numpy(a_t).to(self.args.device)
                # a_t = torch.tensor(a_t, dtype=torch.float32)
                self.model.train()
                optimizer.zero_grad()
                emb, pred_logit, d_logit = self.model(xb, xb_nei)
                emb_s, emb_t = self.model.network_embedding.pairwise_constraint(emb)
                net_pro_loss_s = self.model.network_embedding.net_pro_loss(emb_s, a_s,sample_weight=sample_weight)
                net_pro_loss_t = self.model.network_embedding.net_pro_loss(emb_t, a_t,sample_weight=None)
                net_pro_loss = self.args.net_pro_w * (net_pro_loss_s + net_pro_loss_t)
                if self.args.clf_type == 'multi-class':
                    if sample_weight is not None: ##mask_l 前50为源域样本的 True，后50为目标域样本的False
                        clf_loss = torch.mean(torch.mul(F.cross_entropy(pred_logit[mask_l], torch.argmax(torch.FloatTensor(yb[mask_l]).to(self.args.device), 1), reduction='none'),sample_weight))
                    else:
                        clf_loss = torch.mean(F.cross_entropy(pred_logit[mask_l], torch.argmax(torch.FloatTensor(yb[mask_l]).to(self.args.device), 1), reduction='none'))

                else:
                    # clf_loss = self.clf_loss_f(pred_logit[mask_l], torch.FloatTensor(yb[mask_l]))
                    # clf_loss = torch.sum(clf_loss) / np.sum(mask_l)
                    if sample_weight is not None:
                        clf_loss_f = nn.BCEWithLogitsLoss(reduction='none')
                        half_c_loss =clf_loss_f(pred_logit[mask_l], torch.FloatTensor(yb[mask_l]).to(self.args.device))
                        s=sample_weight.reshape((sample_weight.size()[0],1))
                        a=torch.mul(half_c_loss,s)
                        clf_loss = torch.sum(torch.mul(half_c_loss,s)) / np.sum(mask_l)

                    else:
                        clf_loss_f = nn.BCEWithLogitsLoss(reduction='none')
                        clf_loss = clf_loss_f(pred_logit[mask_l], torch.FloatTensor(yb[mask_l]).to(self.args.device))
                        clf_loss = torch.sum(clf_loss) / np.sum(mask_l)

                # domain_loss = self.domain_loss_f(d_logit, torch.argmax(torch.FloatTensor(domain_label), 1))
                if sample_weight is not None:
                    all_d_loss = F.cross_entropy(d_logit, torch.argmax(torch.FloatTensor(domain_label).to(self.args.device), 1),
                                                 reduction='none')

                    s = sample_weight.size()[0]
                    sample_loss = torch.mul(all_d_loss[:s], sample_weight)
                    domain_loss = torch.mean(torch.cat((sample_loss,all_d_loss[s:]),0))
                    # domain_loss =torch.mean(torch.mul(F.cross_entropy(d_logit, torch.argmax(torch.FloatTensor(domain_label).to(self.args.device), 1), reduction='none'),sample_weight))
                else:
                    domain_loss = torch.mean(F.cross_entropy(d_logit, torch.argmax(torch.FloatTensor(domain_label).to(self.args.device), 1),reduction='none'))

                total_loss = clf_loss + domain_loss + net_pro_loss
                total_loss.backward()
                optimizer.step()
                print('Train Epoch: {}\t Cbatch: {}\t Loss: {:.6f}'.format(cEpoch, cBatch, total_loss.item()))
                if Flag == 'final':
                    with open('DVE final model loss result.txt', 'a', encoding='utf-8') as file:
                        file.write(str(cEpoch) + '\t' + str(cBatch) + '\t' + str(total_loss.item()) + '\n')
                    file.close()
            if Flag=='final':
                self.model.eval()  # deactivates dropout during validation run.
                with torch.no_grad():
                    GradReverse.rate = 1.0
                    _, pred_logit_xs_xt, _ = self.model(whole_xs_xt_stt, whole_xs_xt_stt_nei)
                # save_model_path = "finalmodel/" + str(cEpoch) + "_final_model.ckpt"
                # torch.save(model.state_dict(), save_model_path)

                pred_prob_xs_xt = f.sigmoid(pred_logit_xs_xt) if self.args.clf_type == 'multi-label' else f.softmax(pred_logit_xs_xt)
                pred_prob_xs_xt = pred_prob_xs_xt.cpu().numpy()
                pred_prob_xs = pred_prob_xs_xt[0:num_nodes_s, :]
                pred_prob_xt = pred_prob_xs_xt[-num_nodes_t:, :]
                print('epoch: ', cEpoch)
                f1_s = utils.f1_scores(pred_prob_xs, y_s)
                print('Source micro-F1: %f, macro-F1: %f' % (f1_s[0], f1_s[1]))
                f1_t = utils.f1_scores(pred_prob_xt, y_t)
                print('Target testing micro-F1: %f, macro-F1: %f' % (f1_t[0], f1_t[1]))
                with open('final_model F1-score result.txt', 'a', encoding='utf-8') as file:
                    string = 'epoch: ' + str(cEpoch) + '\n' + \
                             ' Source micro-F1: ' + str(f1_s[0]) + ', macro-F1: ' + str(f1_s[1]) + '\n' + \
                             ' Target testing micro-F1: ' + str(f1_t[0]) + ', macro-F1:' + str(f1_t[1]) + '\n'
                    file.write(string)
                file.close()
                with open('final_model F1-score result(picture).txt', 'a', encoding='utf-8') as file:
                    string = str(cEpoch) + '\t' + str(f1_s[0]) + '\t' + str(f1_s[1]) + '\t' + str(f1_t[0]) + '\t' + str(f1_t[1]) + '\n'
                    file.write(string)
                file.close()
                micro.append(f1_t[0])
                macro.append(f1_t[1])
                if max_micro < f1_t[0]:
                    max_micro = f1_t[0]
                    max_macro = f1_t[1]
                    max_epoch = cEpoch
                    save_model_path = "F1model/" + str(max_epoch) + "_F1_model.ckpt"
                    torch.save(self.model.state_dict(), save_model_path)
                if cEpoch + 1 == epochs:
                    save_model_path = "F1model/" + str(cEpoch) + "_F1_model.ckpt"
                    torch.save(self.model.state_dict(), save_model_path)
        if Flag=='final':
            print('------------------------------------------------')
            print('Flag:', Flag, 'Max_Epoch:', max_epoch, ' max_micro:', max_micro, ' max_macro:', max_macro)
            mean_mi = np.mean(np.array(micro))
            mean_ma = np.mean(np.array(macro))
            print('Flag:', Flag, ' avg_micro:', mean_mi, ' avg_macro:', mean_ma)
            with open('final_model F1-score result.txt', 'a', encoding='utf-8') as file:
                string = 'Max_Epoch:' + str(max_epoch) + ' max_micro:' + str(max_micro) + ', max_macro:' + str(
                    max_macro) + '\n' + \
                         ' avg_micro: ' + str(mean_mi) + ', avg_macro:' + str(mean_ma) + '\n'
                file.write(string)
            file.close()
            print('------------------------------------------------')
    def predict(self, is_train_dataset, batch_size, proba=False):
        self.model.eval()

        whole_xs_xt_stt = self.source_target_dataset.whole_xs_xt_stt
        whole_xs_xt_stt_nei = self.source_target_dataset.whole_xs_xt_stt_nei
        num_nodes_s = self.source_target_dataset.num_nodes_s
        num_nodes_t = self.source_target_dataset.num_nodes_t



        with torch.no_grad():
            GradReverse.rate = 1.0
            emb_s_t, pred_logit_xs_xt, _ = self.model(whole_xs_xt_stt, whole_xs_xt_stt_nei)
        pred_prob_xs_xt = f.sigmoid(pred_logit_xs_xt)if self.args.clf_type == 'multi-label'else f.softmax(pred_logit_xs_xt)
        pred_prob_xs_xt=pred_prob_xs_xt.cpu().numpy()
        emb_s_t=emb_s_t.cpu().numpy()
        pred_prob_xs = pred_prob_xs_xt[0:num_nodes_s, :]
        pred_prob_xt = pred_prob_xs_xt[-num_nodes_t:, :]
        hs = emb_s_t[0:num_nodes_s, :]
        ht = emb_s_t[-num_nodes_t:, :]

        if is_train_dataset:  # for source training
            return pred_prob_xs,hs,ht #概率
        else:
            return pred_prob_xt,hs,ht

        # return pred_prob_xt,hs,ht


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