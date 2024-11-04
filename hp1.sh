python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_acdc_abl --gnn gcn --yaum --scrub_no_kl_2 --scrub_no_kl --scrub_no_kl_combined

python eval_script.py --gnn gcn --dataset Cora --attack_type label --db_name label_acdc_abl --log_name acdc_ablations --start_seed 0 --end_seed 5 --yaum --scrub_no_kl_2 --scrub_no_kl --scrub_no_kl_combined

#####################################

python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_acdc_abl_0.5 --cf 0.5 --gnn gcn --yaum --scrub_no_kl_2 --scrub_no_kl --scrub_no_kl_combined

python eval_script.py --gnn gcn --dataset Cora --attack_type label --db_name label_acdc_abl_0.5 --cf 0.5 --log_name acdc_ablations --start_seed 0 --end_seed 5 --yaum --scrub_no_kl_2 --scrub_no_kl --scrub_no_kl_combined

##########################

python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_acdc_abl_0.25 --cf 0.25 --gnn gcn --scrub_no_kl --scrub_no_kl_combined 

python eval_script.py --gnn gcn --dataset Cora --attack_type label --db_name label_acdc_abl_0.25 --cf 0.25 --log_name acdc_ablations --start_seed 0 --end_seed 5 --scrub_no_kl --scrub_no_kl_combined
