python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_gat_vv_new_cf_0.75 --cf 0.75 --gnn gat --gnndelete

python eval_script.py --gnn gat --dataset Cora --attack_type label --db_name label_gat_vv_new_cf_0.75 --cf 0.75 --log_name newer_gat --start_seed 0 --end_seed 5 --gnndelete

##############################

python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_gat_vv_new_cf_0.05 --cf 0.05 --gnn gat --gnndelete

python eval_script.py --gnn gat --dataset Cora --attack_type label --db_name label_gat_vv_new_cf_0.05 --cf 0.05 --log_name newer_gat --start_seed 0 --end_seed 5 --gnndelete

##############################


python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_gat_vv_new --gnn gat --gnndelete

python eval_script.py --gnn gat --dataset Cora --attack_type label --db_name label_gat_vv_new --log_name newer_gat --start_seed 0 --end_seed 5 --gnndelete