# python run_hp_tune.py --dataset Physics --data_dir /scratch/akshit.sinha/data --df_size 0.5 --random_seed 0 --attack_type label --db_name rebuttal_test_2_new_cf_0.25 --cf 0.25 --gnn gcn --megu --cacdc --gnndelete --scrub

# python eval_script.py --gnn gcn --dataset Physics --data_dir /scratch/akshit.sinha/data --attack_type label --db_name rebuttal_test_2_new_cf_0.25 --cf 0.25 --log_name rebuttal_cf_0.25 --start_seed 0 --end_seed 5 --megu --cacdc --gnndelete --scrub

python run_hp_tune.py --dataset Cora --data_dir /scratch/akshit.sinha/data --df_size 0.5 --random_seed 0 --attack_type label --db_name exp_ks_2 --gnn gcn --cacdc