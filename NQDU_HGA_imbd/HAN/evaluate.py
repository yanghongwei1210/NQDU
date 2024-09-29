import argparse
# from link_prediction import *
from node_classification import *


new_data_folder='E:/HGA(yuan)/Model/HAN/new_data/'
emb_file, record_file = 'emb.dat', 'best_record.dat'




def load(emb_file_path):

    emb_dict = {}
    with open(emb_file_path,'r') as emb_file:        
        for i, line in enumerate(emb_file):
            if i == 0:
                train_para = line[:-1]
            else:
                index, emb = line[:-1].split('\t')
                emb_dict[index] = np.array(emb.split()).astype(np.float32)
        
    return train_para, emb_dict  


def record(args, all_tasks, train_para, all_scores):    

    file_name =  new_data_folder+args.datasetS+'to'+args.datasetT+'_'+record_file
    # with open(f'{data_folder}/{args.datasetT}/{record_file}', 'a') as file:
    with open(file_name, 'a') as file:
        for task, score in zip(all_tasks, all_scores):
            file.write(f'model={args.model}, task={task}, attributed={args.attributed}, supervised={args.supervised}\n')
            file.write(f'{train_para}\n')
            if task=='nc': file.write(f'Macro-F1={score[0]:.4f}, Micro-F1={score[1]:.4f},Accuracy={score[2]:.4f}, Loss={score[3]:.4f}\n')
            elif task=='lp': file.write(f'AUC={score[0]:.4f}, MRR={score[1]:.4f}\n')
            file.write('\n')
    return


def check(args):
    
    if args.attributed=='True':
        if args.model not in ['R-GCN', 'HAN', 'HGT']:
            print(f'{args.model} does not support attributed training!')
            print('Only R-GCN, HAN, and HGT support attributed training!')
            return False
        if args.dataset not in ['DBLP', 'PubMed', 'ACM']:
            print(f'{args.dataset} does not support attributed training!')
            print('Only DBLP and PubMed support attributed training!')
            return False
        
    if args.supervised=='True':
        if args.model not in ['R-GCN', 'HAN', 'HGT']:
            print(f'{args.model} does not support semi-supervised training!')
            print('Only R-GCN, HAN, and HGT support semi-supervised training!')
            return False
        
    return True


