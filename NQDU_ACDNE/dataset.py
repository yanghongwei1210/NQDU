import numpy as np
from torch.utils.data import Dataset
import time
# from HAN.utils import *
from scipy.sparse import lil_matrix
from scipy.sparse import vstack
from ACDNE.utils import *

class DVRLDataset(Dataset):
    def __init__(self, x, y, y_hat):
        self.x = x
        self.y = y
        self.y_hat = y_hat

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.y_hat[idx]

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

    def get_source_data(self):
        # --1-- only features
        # x =lil_matrix.toarray(self.x_s)

        # --2-- features and two adjs
        # x = np.concatenate([lil_matrix.toarray(self.x_s), self.adj_s], axis=1)

        # --3-- feature(or compression) and two PPMI_adj
        x = np.concatenate([lil_matrix.toarray(self.x_s), self.x_n_s], axis=1)

        # --4-- only adj
        # x = self.adj_s

        # --5-- only PPMI_adj
        # x = self.x_n_s

        y = self.y_s
        return x, y
    
    def get_target_data(self):
        # --1-- only features
        # x =lil_matrix.toarray(self.x_t)

        # --2-- features and two adjs
        # x = np.concatenate([lil_matrix.toarray(self.x_t), self.adj_t], axis=1)

        # --3-- feature(or compression) and two PPMI_adj
        x = np.concatenate([lil_matrix.toarray(self.x_t), self.x_n_t], axis=1)

        # --4-- only adj
        # x = self.adj_t

        # --5-- only PPMI_adj
        # x = self.x_n_t

        y = self.y_t

        return x, y


