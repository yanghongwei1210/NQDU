import numpy as np
from torch.utils.data import Dataset
import time
from HAN.utils import *
from scipy.sparse import lil_matrix
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
        self.adjsS, self.id_nameS, self.featuresS = load_data_semisupervisedS(args, args.nodeS, args.linkS, args.configS, list(map(lambda x: int(x), args.metaS.split(','))))
        self.train_poolS, self.train_labelS_np, self.nlabelS, self.multiS = load_label(args.labelS, self.id_nameS)
        self.adjsT, self.id_nameT, self.featuresT = load_data_semisupervisedT(args, args.nodeT, args.linkT, args.configT, list(map(lambda x: int(x), args.metaT.split(','))))
        self.train_poolT, self.train_labelT_np, self.nlabelT, self.multiT = load_label(args.labelT, self.id_nameT)
        # print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 'finish loading', flush=True)
        self.nhead = list(map(lambda x: int(x), args.nhead.split(',')))
        self.nnodeS, self.nchannel, self.nlayer = len(self.id_nameS), len(self.adjsS), len(self.nhead)
        self.nnodeT, self.nchannel, self.nlayer = len(self.id_nameT), len(self.adjsT), len(self.nhead)

        if args.attributed=='True': self.nfeat = self.featuresS.shape[1]
        
        self.embeddingsS = torch.from_numpy(self.featuresS).to(args.device) if args.attributed=='True' else torch.from_numpy(np.random.randn(self.nnodeS, args.size).astype(np.float32)).to(args.device)
        self.embeddingsT = torch.from_numpy(self.featuresT).to(args.device) if args.attributed=='True' else torch.from_numpy(np.random.randn(self.nnodeT, args.size).astype(np.float32)).to(args.device)

        self.train_labelS = torch.from_numpy(self.train_labelS_np.astype(np.float32)).to(args.device)
        # print('train_labelS',train_labelS.shape)
        self.train_labelT = torch.from_numpy(self.train_labelT_np.astype(np.float32)).to(args.device)

    def get_source_data(self):
        # --1-- only features
        # x =self.featuresS

        # --2-- features and two adjs
        tmp = self.featuresS
        meta_path = 'data/' + self.args.datasetS + '/' + self.args.datasetS + '.mat'
        data = sio.loadmat(meta_path)
        adj1 = data['MAM']
        adj2 = data['MDM']
        x = np.concatenate([tmp, adj1, adj2], axis=1)
        # x = np.concatenate([adj1, adj2], axis=1)

        # --3-- feature(or compression) and two PPMI_adj
        # meta_path = 'data/' + self.args.datasetS + '/' + self.args.datasetS + '.mat'
        # data = sio.loadmat(meta_path)
        # adj1 = data['MAM']
        # adj2 = data['MDM']
        # # feature( or compression)
        # X = self.featuresS
        # # X = feature_compression(self.featuresS, dim=1000)
        # Kstep = 3
        # lil_adj1 = lil_matrix(adj1)
        # lil_adj2 = lil_matrix(adj2)
        # A_k_1 = AggTranProbMat(lil_adj1, Kstep)  # ndarray (length,length)
        # A_k_2 = AggTranProbMat(lil_adj2, Kstep)
        # PPMI_1 = ComputePPMI(A_k_1)  # ndarray (length,length)
        # PPMI_2 = ComputePPMI(A_k_2)
        # n_PPMI_1 = MyScaleSimMat(PPMI_1)  # ndarray (length,length)
        # n_PPMI_2 = MyScaleSimMat(PPMI_2)
        # X_n_1 = np.matmul(n_PPMI_1, X)  # neibors' atribute matrix,ndarray (length,length/1000)
        # X_n_2 = np.matmul(n_PPMI_2, X)
        # x = np.concatenate([X, X_n_1, X_n_2], axis=1)


        y = self.train_labelS_np
        return x, y
    
    def get_target_data(self):
        # --1-- only features
        # x = self.featuresT

        # --2-- features and two adjs
        tmp = self.featuresT
        meta_path = 'data/' + self.args.datasetT + '/' + self.args.datasetT + '.mat'
        data = sio.loadmat(meta_path)
        adj1 = data['MAM']
        adj2 = data['MDM']
        x = np.concatenate([tmp, adj1, adj2], axis=1)
        # x = np.concatenate([adj1, adj2], axis=1)

        # --3-- feature(or compression) and two PPMI_adj
        # meta_path = 'data/' + self.args.datasetT + '/' + self.args.datasetT + '.mat'
        # data = sio.loadmat(meta_path)
        # adj1 = data['MAM']
        # adj2 = data['MDM']
        # # feature( or compression)
        # X = self.featuresT
        # # X = feature_compression(self.featuresT, dim=1000)
        #
        # Kstep = 3
        # lil_adj1 = lil_matrix(adj1)
        # lil_adj2 = lil_matrix(adj2)
        # A_k_1 = AggTranProbMat(lil_adj1, Kstep)  # ndarray (length,length)
        # A_k_2 = AggTranProbMat(lil_adj2, Kstep)
        # PPMI_1 = ComputePPMI(A_k_1)  # ndarray (length,length)
        # PPMI_2 = ComputePPMI(A_k_2)
        # n_PPMI_1 = MyScaleSimMat(PPMI_1)  # ndarray (length,length)
        # n_PPMI_2 = MyScaleSimMat(PPMI_2)
        # X_n_1 = np.matmul(n_PPMI_1, X)  # neibors' atribute matrix,ndarray (length,length/1000)
        # X_n_2 = np.matmul(n_PPMI_2, X)
        # x = np.concatenate([X, X_n_1, X_n_2], axis=1)
                
                
        y = self.train_labelT_np
        return x, y


