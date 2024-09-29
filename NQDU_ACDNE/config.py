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
        default=100, #100
        type=int)
    parser.add_argument(
        '--comb_dim',
        help='dimensions of hidden states after combinding with prediction diff',
        default=10, #10
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
        default=10,
        type=int) #1000
    parser.add_argument(
        '--batch_size',
        help='number of batch size for RL',
        default=500,
        type=int) #50000
    parser.add_argument(
        '--learning_rate',
        help='learning rates for RL',
        default=0.0001,
        type=float)
    parser.add_argument(
        '--checkpoint_file_name',
        help='file name for saving and loading the trained model',
        default='tmp/model.ckpt',
        type=str)
    parser.add_argument(
        '--epsilon',
        help='epsilon for RL',
        default=1e-8, # 1e-8
        type=float)
    parser.add_argument(
        '--threshold',
        help='threshold for RL',
        default=0.95, #0.9
        type=float)


    source = "PPI-2" # citationv1,acmv9,dblpv7
    target = 'PPI-1'
    emb_filename = 'emb_'+str(source) + '_' + str(target)
    label_filename = 'label_' + str(source) + '_' + str(target)  # 存放预测的节点标签

    parser.add_argument('--source', default=source, type=str, help='Targeting dataset.',
                        choices=['Blog1', 'Blog2', 'acmv9', 'citationv1', 'dblpv7', 'PPI-1', 'PPI-2', 'PPI-3', 'PPI-4','PPI-5'])
    parser.add_argument('--target', default=target, type=str, help='Targeting dataset.',
                        choices=['Blog1', 'Blog2', 'acmv9', 'citationv1', 'dblpv7', 'PPI-1', 'PPI-2', 'PPI-3', 'PPI-4','PPI-5'])
    parser.add_argument('--seed', type=int, default=0)  # 种子，默认为3
    parser.add_argument('--Kstep', type=int, default=3)
    parser.add_argument('--clf_type', default='multi-label', type=str, help='Targeting task.',
                        choices=['multi-class', 'multi-label'])
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--inner-batch-size', type=int, default=200)
    parser.add_argument('--n_hidden', type=list, default=[512,512])
    parser.add_argument('--n_emb', type=int, default=512)
    parser.add_argument('--l2_w', type=float, default=1e-3)
    parser.add_argument('--lr_ini', type=float, default=0.01)
    parser.add_argument('--net_pro_w', type=float, default=1e-3)
    parser.add_argument('--emb_filename', type=str, default=emb_filename)
    parser.add_argument('--label_filename', type=str, default=label_filename)


    return parser.parse_args()