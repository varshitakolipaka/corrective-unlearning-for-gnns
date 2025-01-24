import os
import argparse

if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='CS')
    argparser.add_argument('--gnn', type=str, default='gcn')
    argparser.add_argument('--data_dir', type=str, default='/scratch/akshit.sinha/data')
    attack_types = ['label']
    
    args = argparser.parse_args()
    
    os.system(f"python hp_tune_copy.py --dataset {args.dataset} --gnn {args.gnn}")
    os.system(f"python get_base_model_hps.py --dataset {args.dataset} --gnn {args.gnn}")
    for attack_type in attack_types:
        os.system(f"python train_and_save_gnn.py --dataset {args.dataset} --gnn {args.gnn} --attack_type {attack_type} --data_dir {args.data_dir}")
    