import numpy as np
import scipy.io as sio
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
import scipy
from sklearn.metrics import f1_score,accuracy_score
import scipy.sparse as sp
from sklearn.decomposition import PCA
from warnings import filterwarnings
import random
import torch
filterwarnings('ignore')


def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

def load_network(file):
    net = sio.loadmat(file)
    x, a, y = net['attrb'], net['network'], net['group']
    if not isinstance(x, scipy.sparse.lil_matrix):
        x = lil_matrix(x)
    if not isinstance(a, scipy.sparse.csc_matrix):
        a = csc_matrix(a)
    return a, x, y


def feature_compression(features, dim=200):
    """Preprcessing of features"""
    features = features.toarray()
    feat = lil_matrix(PCA(n_components=dim, random_state=0).fit_transform(features))
    return feat

def feature_compression_ppi(features, dim=200):
    """Preprcessing of features"""
    features = features.toarray()
    feat = lil_matrix(features)
    return feat

def my_scale_sim_mat(w):
    """L1 row norm of a matrix"""
    rowsum = np.array(np.sum(w, axis=1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    w = r_mat_inv.dot(w)
    return w


def agg_tran_prob_mat(g, step):
    """aggregated K-step transition probality"""
    g = my_scale_sim_mat(g)
    g = csc_matrix.toarray(g)
    a_k = g
    a = g
    for k in np.arange(2, step+1):
        a_k = np.matmul(a_k, g)
        a = a+a_k/k
    return a


def compute_ppmi(a):
    """compute PPMI, given aggregated K-step transition probality matrix as input"""
    np.fill_diagonal(a, 0)
    a = my_scale_sim_mat(a)
    (p, q) = np.shape(a)
    col = np.sum(a, axis=0)
    col[col == 0] = 1
    ppmi = np.log((float(p)*a)/col[None, :])
    idx_nan = np.isnan(ppmi)
    ppmi[idx_nan] = 0
    ppmi[ppmi < 0] = 0
    return ppmi


def shuffle_aligned_list(data):
    num = data[0].shape[0]
    shuffle_index = np.random.permutation(num)
    return shuffle_index, [d[shuffle_index] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    shuffle_index = None
    if shuffle:
        shuffle_index, data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= data[0].shape[0]:
            batch_count = 0
            if shuffle:
                shuffle_index, data = shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data], shuffle_index[start:end]


def batch_ppmi(batch_size, shuffle_index_s, shuffle_index_t, ppmi_s, ppmi_t):
    """return the PPMI matrix between nodes in each batch"""
    # #proximity matrix between source network nodes in each mini-batch
    # noinspection DuplicatedCode
    a_s = np.zeros((int(batch_size / 2), int(batch_size / 2)))
    for ii in range(int(batch_size / 2)):
        for jj in range(int(batch_size / 2)):
            if ii != jj:
                a_s[ii, jj] = ppmi_s[shuffle_index_s[ii], shuffle_index_s[jj]]
    # #proximity matrix between target network nodes in each mini-batch
    # noinspection DuplicatedCode
    a_t = np.zeros((int(batch_size / 2), int(batch_size / 2)))
    for ii in range(int(batch_size / 2)):
        for jj in range(int(batch_size / 2)):
            if ii != jj:
                a_t[ii, jj] = ppmi_t[shuffle_index_t[ii], shuffle_index_t[jj]]
    return my_scale_sim_mat(a_s), my_scale_sim_mat(a_t)


def f1_scores(y_pred, y_true):
    def predict(y_tru, y_pre):
        top_k_list = np.array(np.sum(y_tru, 1), np.int32)
        prediction = []
        for i in range(y_tru.shape[0]):
            pred_i = np.zeros(y_tru.shape[1])
            pred_i[np.argsort(y_pre[i, :])[-top_k_list[i]:]] = 1
            prediction.append(np.reshape(pred_i, (1, -1)))
        prediction = np.concatenate(prediction, axis=0)
        return np.array(prediction, np.int32)
    results = {}
    predictions = predict(y_true, y_pred)
    averages = ["micro", "macro"]
    for average in averages:
        results[average] = f1_score(y_true, predictions, average=average)
    return results["micro"], results["macro"]

def accuracy(y_pred, y_true):
    # 参数：模型预测的标签和真实标签
    def predict(y_true, y_pred):
        top_k_list = np.array(np.sum(y_true, 1), np.int32)  # 标签按行求和，得到每个节点含标签数目
        predictions = []
        for i in range(y_true.shape[0]):  # 遍历每一个节点
            pred_i = np.zeros(y_true.shape[1])  # 全0 shape(3,) 初始化每个节点的标签为0
            pred_i[np.argsort(y_pred[i, :])[-top_k_list[i]:]] = 1
            # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y。
            #y_pred[i, :]取第i个节点的预测标签，
            #np.argsort(y_pred[i, :])将该节点的预测标签概率从小到大排列，返回从小到大的索引，排在后面的索引，其对应标签为节点标签的可能性大
            #[-top_k_list[i]:]倒着数（每个节点含标签数目）个元素
            #np.argsort(y_pred[i, :])[-top_k_list[i]:] 从np.argsort(y_pred[i, :])中倒着取top_k_list[i]个元素，得到可能性大的top_k_list[i]个标签的索引
            #pred_i[索引]=1，将可能性大的索引对应的标签赋值为 1
            predictions.append(np.reshape(pred_i, (1, -1))) #将该节点的标签格式由(3,)->(1,3)
        predictions = np.concatenate(predictions, axis=0)
        #numpy.concatenate((a1,a2,...), axis=0)函数。能够一次完成多个数组的拼接。其中a1,a2,...是数组类型的参数
        #a1(1*3),a2(1*3)->拼接后(1*6)
        return np.array(predictions, np.int32)

    predictions = predict(y_true, y_pred)
    _accuracy=accuracy_score(y_true, predictions)
    return _accuracy

from scipy.sparse import vstack
def load_source_target_data(args):

    ####################
    # Load source data
    ####################
    A_s, X_s, Y_s = load_network('data/' + str(args.source) + '.mat')
    num_nodes_S = X_s.shape[0]
    ####################
    # Load target data
    ####################
    A_t, X_t, Y_t = load_network('data/' + str(args.target) + '.mat')
    num_nodes_T = X_t.shape[0]
    features = vstack((X_s, X_t))
    if args.source in ['Blog1', 'Blog2'] or args.target in ['Blog1', 'Blog2']:
        features = feature_compression(features, dim=1000)
    else:
        features = feature_compression_ppi(features, dim=1000)
    X_s = features[0:num_nodes_S, :]
    X_t = features[-num_nodes_T:, :]
    '''compute PPMI'''
    A_k_s = agg_tran_prob_mat(A_s, args.Kstep)
    PPMI_s = compute_ppmi(A_k_s)
    n_PPMI_s = my_scale_sim_mat(PPMI_s)  # row normalized PPMI
    X_n_s = np.matmul(n_PPMI_s, lil_matrix.toarray(X_s))  # neibors' attribute matrix
    # noinspection DuplicatedCode
    '''compute PPMI'''
    A_k_t = agg_tran_prob_mat(A_t, args.Kstep)
    PPMI_t = compute_ppmi(A_k_t)
    n_PPMI_t = my_scale_sim_mat(PPMI_t)  # row normalized PPMI
    X_n_t = np.matmul(n_PPMI_t, lil_matrix.toarray(X_t))  # neibors' attribute matrix
    # #input data
    input_data = dict()
    input_data['adj_S'] = csc_matrix.toarray((A_s))
    input_data['adj_T'] = csc_matrix.toarray((A_t))
    input_data['PPMI_S'] = PPMI_s
    input_data['PPMI_T'] = PPMI_t
    input_data['attrb_S'] = X_s  #压缩后的特征
    input_data['attrb_T'] = X_t
    input_data['attrb_nei_S'] = X_n_s
    input_data['attrb_nei_T'] = X_n_t
    input_data['label_S'] = Y_s
    input_data['label_T'] = Y_t
    return input_data