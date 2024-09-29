import argparse
import torch

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--problem',
        choices=['regression', 'classification'],
        help='problem type',
        default='classification',
        type=str)
    parser.add_argument(
        '--device',
        help='train device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        type=str)
    parser.add_argument(
        '--normalization',
        choices=['minmax', 'standard'],
        help='data normalization method',
        default='minmax',
        type=str)
    parser.add_argument(
        '--hidden_dim',
        help='dimensions of hidden states',
        default=100,
        type=int)
    parser.add_argument(
        '--comb_dim',
        help='dimensions of hidden states after combinding with prediction diff',
        default=10,
        type=int)
    parser.add_argument(
        '--layer_number',
        help='number of network layers',
        default=5,
        type=int)
    parser.add_argument(
        '--act_fn',
        help='activation function',
        default='Tanh',
        type=str)
    parser.add_argument(
        '--outer_iterations',
        help='number of iterations',
        default=1,
        type=int) #1000
    parser.add_argument(
        '--batch_size',
        help='number of batch size for RL',
        default=500,
        type=int) #50000
    parser.add_argument(
        '--learning_rate',
        help='learning rates for RL',
        default=0.001,
        type=float)
    parser.add_argument(
        '--checkpoint_file_name',
        help='file name for saving and loading the trained model',
        default='tmp/model.ckpt',
        type=str)
    parser.add_argument(
        '--epsilon',
        help='epsilon for RL',
        default=1e-8,
        type=float)
    parser.add_argument(
        '--threshold',
        help='threshold for RL',
        default=0.9,
        type=float)


    datasetS = "imbd_b"
    folderS = "data/"+datasetS+"/"
    node_fileS = folderS+"node.dat" #源节点原始特征
    config_fileS = folderS+"config.dat"
    link_fileS = folderS+"link.dat" #源节点边（按元路径划分）
    label_fileS = folderS+"labelall.dat" #源节点标签

    ## gaizheli
    emb_fileS = 'E:/DVRL_HAN/new_data/' + datasetS + "/emb.dat"

    datasetT = "imbd_a"
    folderT = "data/"+datasetT+"/"
    node_fileT = folderT+"node.dat" #目标节点原始特征
    config_fileT = folderT+"config.dat"
    link_fileT = folderT+"link.dat" #目标节点边（按元路径划分）
    label_fileT = folderT+"labelall.dat" #目标节点标签

    ## gaizheli
    emb_fileT='E:/DVRL_HAN/new_data/' + datasetT + "/emb.dat"

    metaS = "1,2,3"
    metaT = "1,2,3"
    ##源域
    parser.add_argument('--nodeS', type=str, default=node_fileS) #节点
    parser.add_argument('--linkS', type=str, default=link_fileS) #边
    parser.add_argument('--configS', type=str,default=config_fileS)
    parser.add_argument('--labelS', type=str, default=label_fileS) #标签
    parser.add_argument('--outputS', type=str, default=emb_fileS) #输出
    parser.add_argument('--metaS', type=str, default=metaS) #选择用于训练的元路径
    ##目标域
    parser.add_argument('--nodeT', type=str, default=node_fileT) #节点
    parser.add_argument('--linkT', type=str, default=link_fileT) #边
    parser.add_argument('--configT', type=str, default=config_fileT)
    parser.add_argument('--labelT', type=str, default=label_fileT) #标签
    parser.add_argument('--outputT', type=str, default=emb_fileT) #输出
    parser.add_argument('--metaT', type=str,default=metaT) #元路径

    parser.add_argument('--seed', type=int, default=3) #种子，默认为3
    # parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--size', type=int, default=64) #50
    parser.add_argument('--nhead', type=str, default='8')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--dropout', type=float, default=0.4)

    parser.add_argument('--neigh-por', type=float, default=0.6)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--inner-batch-size', type=int, default=128) #256
    parser.add_argument('--epochs', type=int, default=10)  #500

    parser.add_argument('--attributed', type=str, default="True")
    parser.add_argument('--supervised', type=str, default="True")

    parser.add_argument('--datasetS', default=datasetS, type=str, help='Targeting dataset.',
                        choices=['imbd_a','imbd_b','imbd_c'])
    parser.add_argument('--datasetT', default=datasetT, type=str, help='Targeting dataset.',
                        choices=['imbd_a','imbd_b','imbd_c'])
    parser.add_argument('--model', default='HAN', type=str, help='Targeting model.',
                        choices=['metapath2vec-ESim','PTE','HIN2Vec','AspEm','HEER','R-GCN','HAN','HGT','TransE','DistMult', 'ConvE'])
    parser.add_argument('--task', default='nc', type=str, help='Targeting task.',
                        choices=['nc', 'lp', 'both'])

    return parser.parse_args()