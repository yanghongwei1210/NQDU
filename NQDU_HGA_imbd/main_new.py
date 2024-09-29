import os
import numpy as np

from dataset import GraphSourceTargetDataset
from config import parse_arguments
from dvrl_han import DVRL_HAN
from HAN import HAN_Trainer


def main(args, num):
    print('num:',num)
    print('*source:' + args.datasetS + ' target:' + args.datasetT)
    # Data loading
    # The number of training and validation samples
    print('Start Data loading......')
    dataset = GraphSourceTargetDataset(args)
    x_source, y_source = dataset.get_source_data()
    x_valid, y_valid = dataset.get_target_data()
    x_target, y_target = dataset.get_target_data()
    print('Finished data preprocess.')
    print('**************************')

    # Predictor model definition
    print('Start Predictor model definition......')
    pred_model = HAN_Trainer(args, dataset)
    print('Finished Predictor model definition.')
    print('**************************')
    # Flags for using stochastic gradient descent / pre-trained model
    flags = {'sgd': False, 'pretrain': False}

    # Initializes DVRL
    print('Start Initializes DVRL......')
    dvrl = DVRL_HAN(args, pred_model, flags)
    print('Finished Initializes DVRL.')
    print('**************************')

    print('Start train DVE......')
    dvrl.train_dvrl('accuracy',num)
    # dvrl.train_dvrl('mi-f1', num)
    print('Finished dvrl training.')
    print('**************************')



if __name__ == '__main__':
    args = parse_arguments()
    num = 1
    end_acc = 0
    end_micro = 0
    end_macro = 0
    avg_acc=0
    avg_micro=0
    avg_macro=0
    main(args, num)




