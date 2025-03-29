import os
import argparse

if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='Cora')
    argparser.add_argument('--gnn', type=str, default='gcn')
    argparser.add_argument('--data_dir', type=str, default='/scratch/akshit.sinha/data/sumit')
    argparser.add_argument('--train_oracle', action='store_true')
    argparser.add_argument('--df_size', type=float, default=0.5)
    argparser.add_argument('--attack_type', type=str, default='label')

    args = argparser.parse_args()

    # if args.train_oracle:
    #     os.system(f"python hp_tune_copy.py --dataset {args.dataset} --gnn {args.gnn} --attack_type {args.attack_type}")
    os.system(f"python get_base_model_hps.py --dataset {args.dataset} --gnn {args.gnn} --attack_type {args.attack_type}")
    cmd = f"python train_and_save_gnn.py --dataset {args.dataset} --gnn {args.gnn} --attack_type {args.attack_type} --data_dir {args.data_dir} --df_size {args.df_size}"
    if args.train_oracle:
        cmd += " --train_oracle"
    os.system(cmd)