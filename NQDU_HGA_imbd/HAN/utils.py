import gc
import random
import numpy as np
from collections import defaultdict

import torch


def convert(posi, nega, posi_size, nega_size, batch_size):
    
    posi = posi[np.random.randint(posi_size, size=batch_size), :]
    nega = nega[np.random.randint(nega_size, size=batch_size), :]
    
    seeds = set()
    for each in posi.flatten():
        seeds.add(each)
    for each in nega.flatten():
        seeds.add(each)
    seeds = np.sort(list(seeds))
    
    index_dict = {k:v for v,k in enumerate(seeds)}
    indices = np.array([index_dict[k] for k in seeds])
    
    new_posi, new_nega = [], []
    for (pleft, pright), (nleft, nright) in zip(posi, nega):
        new_posi.append([index_dict[pleft], index_dict[pright]])
        new_nega.append([index_dict[nleft], index_dict[nright]])
    
    return seeds, indices, np.array(new_posi), np.array(new_nega)
        

def set_seed(seed, device):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device=="cuda":
        torch.cuda.manual_seed(seed)


def sample(target_pool, positive_edges):
    
    positive_pool = set()
    for edge in positive_edges:
        positive_pool.add(tuple(sorted(edge)))
        
    negative_edges = set()
    positive_count, negative_count = positive_edges.shape[0], 0
    while negative_count < positive_count:
        nega_left, nega_right = np.random.choice(list(target_pool), size=positive_count-int(negative_count/2), replace=True), np.random.choice(list(target_pool), size=positive_count-int(negative_count/2), replace=True)
        for each_left, each_right in zip(nega_left, nega_right):
            if each_left==each_right: continue
            if tuple(sorted([each_left, each_right])) in positive_pool: continue
            if (each_left, each_right) in negative_edges: continue
            negative_edges.add((each_left, each_right))
            negative_count += 1
            if negative_count >= positive_count: break
                
    negative_edges = np.array(list(negative_edges)).astype(np.int32)
    return negative_edges    


def load_data_unsupervised(args, node, edge, config, meta):
    print('check 0', flush=True)
    lines = open(config, 'r').readlines()
    target, positive_type = int(lines[0][:-1]), int(lines[1][:-1])
    useful_types, positive_same, positive_edges = set(), False, []
    for each in lines[2].split('\t'):
        start, end, ltype = each.split(',')
        start, end, ltype = int(start), int(end), int(ltype)
        if ltype in meta:
            useful_types.add(ltype)
        if ltype==positive_type and start==target and end==target:
            positive_same = True
    print('check 1', flush=True)
    id_inc, id_name, name_id, name_attr = 0, {}, {}, {}
    with open(node, 'r') as file:
        for line in file:
            if args.attributed=='True': nid, ntype, attr = line[:-1].split('\t')
            elif args.attributed=='False': nid, ntype = line[:-1].split('\t')
            nid, ntype = int(nid), int(ntype)
            if ntype==target:
                name_id[nid] = id_inc
                id_name[id_inc] = nid
                if args.attributed=='True': name_attr[nid] = np.array(attr.split(',')).astype(np.float32)
                id_inc += 1
    print('check 2', flush=True)
    type_corners = {ltype:defaultdict(set) for ltype in useful_types}
    with open(edge, 'r') as file:
        for line in file:
            start, end, ltype = line[:-1].split('\t')
            start, end, ltype = int(start), int(end), int(ltype)
            if ltype in useful_types:
                if start in name_id:
                    type_corners[ltype][end].add(name_id[start])
                if end in name_id:
                    type_corners[ltype][start].add(name_id[end])
            if ltype==positive_type and positive_same:
                positive_edges.append([name_id[start], name_id[end]])
        if positive_same:
            positive_edges = np.array(positive_edges).astype(np.int32)
    
    print('check 3', flush=True)            
    adjs = []
    for ltype in useful_types:
        corners = type_corners[ltype]
        two_hops = defaultdict(set)
        for _, neighbors in corners.items():
            for snode in neighbors:
                for enode in neighbors:
                    if snode!=enode:
                        two_hops[snode].add(enode)
        print('check 3.1', flush=True)
        rights, counts = [], np.zeros(id_inc).astype(int)
        for i in range(id_inc):
            if i in two_hops:
                current = np.sort(list(two_hops[i]))
                rights.append(current)
                counts[i] = len(current)
        adjs.append((np.concatenate(rights), counts))  
        print('check 3.2', flush=True)
        if ltype==positive_type and not positive_same:
            for _, neighbors in corners.items():
                for snode in neighbors:
                    for enode in neighbors:
                        positive_edges.append([snode, enode])
            positive_edges = np.array(positive_edges).astype(np.int32)
        del two_hops, rights, counts, type_corners[ltype]
        gc.collect()
        print('check 3.3', flush=True)
    print('check 4', flush=True)
    
    if args.attributed=="True": name_attr = np.array([name_attr[id_name[i]] for i in range(len(id_name))]).astype(np.float32)
    
    return adjs, id_name, set(range(id_inc)), positive_edges, name_attr


def load_label(label_path, id_name):
    
    name_id, id_label, all_labels = {v:k for k,v in id_name.items()}, {}, set()
    # print(name_id)#在node中的位置：在node1中的位置
    
    train_set, multi = set(), False
    with open(label_path, 'r') as file:
        for line in file:
            node, label = line[:-1].split('\t')
            train_set.add(name_id[int(node)])#有标签的节点在node1中的位置
            if multi or ',' in label:
                multi = True
                label_array = np.array(label.split(',')).astype(int)
                for each in label_array:
                    all_labels.add(each)
                id_label[name_id[int(node)]] = label_array#有标签的节点在node1中的位置：标签
            else:
                all_labels.add(int(label))
                id_label[name_id[int(node)]] = int(label)            
    train_pool = np.sort(list(train_set))
    
    train_label = []
    for k in train_pool:
        if multi:
            curr_label = np.zeros(len(all_labels)).astype(int)
            curr_label[id_label[k]] = 1
            train_label.append(curr_label)
        else:
            train_label.append(id_label[k])
    train_label = np.array(train_label)
##有标签的节点在node1中的位置、对应的标签、标签类别个数
    return train_pool, train_label, len(all_labels), multi


import scipy.io as sio
def load_data_semisupervisedS(args, node, edge, config, meta):
    # args, args.nodeS 参数，源节点原始特征
    print('check 1', flush=True)
    id_inc, id_name, name_id, name_attr = 0, {}, {}, {}
    # id_inc 节点数量
    with open(node, 'r') as file:
        for line in file:  # 每一个节点的原始特征
            if args.attributed == 'True':
                nid, ntype, attr = line[:-1].split('\t')
            # 节点id 节点type 节点原始特征
            elif args.attributed == 'False':
                nid, ntype = line[:-1].split('\t')
            nid, ntype = int(nid), int(ntype)
            if ntype == 1:
                name_id[nid] = id_inc
                # 每一个节点的索引
                # （如果第一个节点的id=0,则 nid=0 ,name_id[0]=0 ）
                # （如果第二个节点的id=14,则 nid=14 , name_id[14] = 1）
                id_name[id_inc] = nid
                # 每一个索引对应的节点id
                # ( 索引为 0 对应的节点为 nid=0 ; 索引为 1 对应的节点为 nid=14)
                if args.attributed == 'True': name_attr[nid] = np.array(attr.split(',')).astype(np.float32)
                # 每一个节点的原始特征
                id_inc += 1

    meta_path='data/' + args.datasetS + '/'+args.datasetS+'.mat'
    data = sio.loadmat(meta_path)
    meta1_mam=data['MAM']
    meta2_mdm=data['MDM']

    adjs = []
    mam_len1=meta1_mam.shape[0]
    mam_len2 = meta1_mam.shape[1]
    two_hops_1 = defaultdict(set)
    for snode in range(mam_len1):
        for enode in range(mam_len2):
            if snode != enode and meta1_mam[snode][enode]==1:
                two_hops_1[snode].add(enode)

    rights_1, counts_1 = [], np.zeros(id_inc).astype(int)
    for i in range(id_inc):
        if i in two_hops_1:
            current = np.sort(list(two_hops_1[i]))
            # print('current',current)
            rights_1.append(current)
            # print('rights',rights)
            counts_1[i] = len(current)
            # print('counts',counts)
    adjs.append((np.concatenate(rights_1), counts_1))

    mdm_len1 = meta2_mdm.shape[0]
    mdm_len2 = meta2_mdm.shape[1]
    two_hops_2 = defaultdict(set)
    for snode in range(mdm_len1):
        for enode in range(mdm_len2):
            if snode != enode and meta2_mdm[snode][enode] == 1:
                two_hops_2[snode].add(enode)
    rights_2, counts_2 = [], np.zeros(id_inc).astype(int)
    for i in range(id_inc):
        if i in two_hops_2:
            current = np.sort(list(two_hops_2[i]))
            # print('current',current)
            rights_2.append(current)
            # print('rights',rights)
            counts_2[i] = len(current)
            # print('counts',counts)
    adjs.append((np.concatenate(rights_2), counts_2))

    if args.attributed == "True": name_attr = np.array([name_attr[id_name[i]] for i in range(len(id_name))]).astype(
        np.float32)
    return adjs, id_name, name_attr


def load_data_semisupervisedT(args, node, edge, config, meta):
    # args, args.nodeS 参数，源节点原始特征
    print('check 1', flush=True)
    id_inc, id_name, name_id, name_attr = 0, {}, {}, {}
    # id_inc 节点数量
    with open(node, 'r') as file:
        for line in file:  # 每一个节点的原始特征
            if args.attributed == 'True':
                nid, ntype, attr = line[:-1].split('\t')
            # 节点id 节点type 节点原始特征
            elif args.attributed == 'False':
                nid, ntype = line[:-1].split('\t')
            nid, ntype = int(nid), int(ntype)
            if ntype == 1:
                name_id[nid] = id_inc
                # 每一个节点的索引
                # （如果第一个节点的id=0,则 nid=0 ,name_id[0]=0 ）
                # （如果第二个节点的id=14,则 nid=14 , name_id[14] = 1）
                id_name[id_inc] = nid
                # 每一个索引对应的节点id
                # ( 索引为 0 对应的节点为 nid=0 ; 索引为 1 对应的节点为 nid=14)
                if args.attributed == 'True': name_attr[nid] = np.array(attr.split(',')).astype(np.float32)
                # 每一个节点的原始特征
                id_inc += 1

    meta_path='data/' + args.datasetT + '/'+args.datasetT+'.mat'
    data = sio.loadmat(meta_path)
    meta1_mam=data['MAM']
    meta2_mdm=data['MDM']

    adjs = []
    mam_len1=meta1_mam.shape[0]
    mam_len2 = meta1_mam.shape[1]
    two_hops_1 = defaultdict(set)
    for snode in range(mam_len1):
        for enode in range(mam_len2):
            if snode != enode and meta1_mam[snode][enode]==1:
                two_hops_1[snode].add(enode)

    rights_1, counts_1 = [], np.zeros(id_inc).astype(int)
    for i in range(id_inc):
        if i in two_hops_1:
            current = np.sort(list(two_hops_1[i]))
            # print('current',current)
            rights_1.append(current)
            # print('rights',rights)
            counts_1[i] = len(current)
            # print('counts',counts)
    adjs.append((np.concatenate(rights_1), counts_1))

    mdm_len1 = meta2_mdm.shape[0]
    mdm_len2 = meta2_mdm.shape[1]
    two_hops_2 = defaultdict(set)
    for snode in range(mdm_len1):
        for enode in range(mdm_len2):
            if snode != enode and meta2_mdm[snode][enode] == 1:
                two_hops_2[snode].add(enode)
    rights_2, counts_2 = [], np.zeros(id_inc).astype(int)
    for i in range(id_inc):
        if i in two_hops_2:
            current = np.sort(list(two_hops_2[i]))
            # print('current',current)
            rights_2.append(current)
            # print('rights',rights)
            counts_2[i] = len(current)
            # print('counts',counts)
    adjs.append((np.concatenate(rights_2), counts_2))

    if args.attributed == "True": name_attr = np.array([name_attr[id_name[i]] for i in range(len(id_name))]).astype(
        np.float32)
    return adjs, id_name, name_attr



import scipy.sparse as sp
from scipy.sparse import csc_matrix
from sklearn.decomposition import PCA

def MyScaleSimMat(W):
    '''L1 row norm of a matrix'''
    rowsum = np.array(np.sum(W, axis=1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    W = r_mat_inv.dot(W)
    return W


def AggTranProbMat(G, step):
    '''aggregated K-step transition probality'''
    G = MyScaleSimMat(G)
    G = csc_matrix.toarray(G)
    A_k = G
    A = G
    for k in np.arange(2, step + 1):
        A_k = np.matmul(A_k, G)
        A = A + A_k / k

    return A


def ComputePPMI(A):
    '''compute PPMI, given aggregated K-step transition probality matrix as input'''
    np.fill_diagonal(A, 0)
    A = MyScaleSimMat(A)
    (p, q) = np.shape(A)
    col = np.sum(A, axis=0)
    col[col == 0] = 1
    PPMI = np.log((float(p) * A) / col[None, :])
    IdxNan = np.isnan(PPMI)
    PPMI[IdxNan] = 0
    PPMI[PPMI < 0] = 0

    return PPMI

def feature_compression(features, dim=200):
    """Preprcessing of features"""
    feat = PCA(n_components=dim, random_state=0).fit_transform(features)
    return feat

from sklearn.metrics import f1_score
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