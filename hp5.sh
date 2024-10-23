python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_gat_vv_new_cf_0.5 --cf 0.5 --gnn gat --cacdc --contra_2 --yaum --retrain --gif --utu --scrub --megu

python eval_script.py --gnn gat --dataset Cora --attack_type label --db_name label_gat_vv_new_cf_0.5 --cf 0.5 --log_name newer_gat --start_seed 0 --end_seed 5 --cacdc --contra_2 --yaum --retrain --gif --utu --scrub --megu

##############################

python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_gat_vv_new_cf_0.25 --cf 0.25 --gnn gat --cacdc --contra_2 --yaum --retrain --gif --utu --scrub --megu

python eval_script.py --gnn gat --dataset Cora --attack_type label --db_name label_gat_vv_new_cf_0.25 --cf 0.25 --log_name newer_gat --start_seed 0 --end_seed 5 --cacdc --contra_2 --yaum --retrain --gif --utu --scrub --megu

##############################