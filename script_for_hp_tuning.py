import os
import argparse

"""
Example command:
python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_acdc_abl_0.75 --cf 0.75 --gnn gcn --yaum --scrub_no_kl_2

python eval_script.py --dataset CS --attack_type edge --df_size 3000 --start_seed 0 --end_seed 5 --db_name edge_main --log_name edge_logs --scrub --gif
"""
 
if __name__=="__main__":
    corrective_fractions = [0.05, 0.25, 0.5, 0.75, 1][::-1]

    start_seed, end_seed = 0, 5
    db_name_prefix = "icml_runs"
    log_name = "icml_runs"
    
    parser = argparse.ArgumentParser(description="Run HP tuning for various unlearning models")

    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--df_size', type=float, default=0.5, help='Size of the dataset fraction')
    parser.add_argument('--data_dir', type=str, default='/scratch/akshit.sinha/data', help='Directory containing the data')
    
    args = parser.parse_args()
    
    methods = [
        'utu',
        'retrain',
        'scrub',
        'megu',
        'gnndelete',
        'gif',
        'yaum',
        'cacdc',
    ]
    
    methods = [f'--{method}' for method in methods]

    for cf in corrective_fractions:
        cmd = f"python run_hp_tune.py --dataset {args.dataset} --df_size {args.df_size} --random_seed {start_seed} --data_dir {args.data_dir} --attack_type label"
        
        if cf != 1:
            db_name = f"{db_name_prefix}_{cf}"
        else:
            db_name = db_name_prefix
            
        cmd += f" --db_name {db_name} --cf {cf} --gnn gcn"
        
        for method in methods:
            cmd += f" {method}"
            
        print(f"Running command: {cmd}")
        
        os.system(cmd)
        
        ####
        
        cmd = f"python eval_script.py --dataset {args.dataset} --df_size {args.df_size} --start_seed {start_seed} --end_seed {end_seed} --data_dir {args.data_dir} --db_name {db_name} --log_name {log_name} --gnn gcn --cf {cf}"
        
        for method in methods:
            cmd += f" {method}"
            
        print(f"Running command: {cmd}")
        
        os.system(cmd)
    
        os.system(f"sh get_stats.sh logs/{log_name}")