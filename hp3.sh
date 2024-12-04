python run_hp_tune.py --dataset Physics --data_dir /scratch/akshit.sinha/data --df_size 0.5 --random_seed 0 --attack_type label --db_name rebuttal_test_new --gnn gcn --megu

python eval_script.py --gnn gcn --dataset Physics --data_dir /scratch/akshit.sinha/data --attack_type label --db_name rebuttal_test_new --log_name rebuttal --start_seed 0 --end_seed 5 --megu

#######################

python run_hp_tune.py --dataset Computers --data_dir /scratch/akshit.sinha/data --df_size 0.5 --random_seed 0 --attack_type label --db_name rebuttal_test_new --gnn gcn --megu

python eval_script.py --gnn gcn --dataset Computers --data_dir /scratch/akshit.sinha/data --attack_type label --db_name rebuttal_test_new --log_name rebuttal --start_seed 0 --end_seed 5 --megu

#######################

python run_hp_tune.py --dataset DBLP --data_dir /scratch/akshit.sinha/data --df_size 0.5 --random_seed 0 --attack_type label --db_name rebuttal_test_new --gnn gcn --megu --cacdc --gnndelete --scrub

python eval_script.py --gnn gcn --dataset DBLP --data_dir /scratch/akshit.sinha/data --attack_type label --db_name rebuttal_test_new --log_name rebuttal --start_seed 0 --end_seed 5 --megu --cacdc --gnndelete --scrub

#######################

python run_hp_tune.py --dataset Physics --data_dir /scratch/akshit.sinha/data --df_size 0.5 --random_seed 0 --attack_type label --db_name rebuttal_test_new_cf_0.05 --cf 0.05 --gnn gcn --megu --cacdc --gnndelete --scrub

python eval_script.py --gnn gcn --dataset Physics --data_dir /scratch/akshit.sinha/data --attack_type label --db_name rebuttal_test_new_cf_0.05 --cf 0.05 --log_name rebuttal_cf_0.05 --start_seed 0 --end_seed 5 --megu --cacdc --gnndelete --scrub

#######################

python run_hp_tune.py --dataset Computers --data_dir /scratch/akshit.sinha/data --df_size 0.5 --random_seed 0 --attack_type label --db_name rebuttal_test_new_cf_0.05 --cf 0.05 --gnn gcn --megu --cacdc --gnndelete --scrub

python eval_script.py --gnn gcn --dataset Computers --data_dir /scratch/akshit.sinha/data --attack_type label --db_name rebuttal_test_new_cf_0.05 --cf 0.05 --log_name rebuttal_cf_0.05 --start_seed 0 --end_seed 5 --megu --cacdc --gnndelete --scrub

#######################

python run_hp_tune.py --dataset DBLP --data_dir /scratch/akshit.sinha/data --df_size 0.5 --random_seed 0 --attack_type label --db_name rebuttal_test_new_cf_0.05 --cf 0.05 --gnn gcn --megu --cacdc --gnndelete --scrub

python eval_script.py --gnn gcn --dataset DBLP --data_dir /scratch/akshit.sinha/data --attack_type label --db_name rebuttal_test_new_cf_0.05 --cf 0.05 --log_name rebuttal_cf_0.05 --start_seed 0 --end_seed 5 --megu --cacdc --gnndelete --scrub

#######################
#######################

python run_hp_tune.py --dataset Physics --data_dir /scratch/akshit.sinha/data --df_size 0.5 --random_seed 0 --attack_type label --db_name rebuttal_test_new_cf_0.5 --cf 0.5 --gnn gcn --megu --cacdc --gnndelete --scrub

python eval_script.py --gnn gcn --dataset Physics --data_dir /scratch/akshit.sinha/data --attack_type label --db_name rebuttal_test_new_cf_0.5 --cf 0.5 --log_name rebuttal_cf_0.5 --start_seed 0 --end_seed 5 --megu --cacdc --gnndelete --scrub

#######################

python run_hp_tune.py --dataset Computers --data_dir /scratch/akshit.sinha/data --df_size 0.5 --random_seed 0 --attack_type label --db_name rebuttal_test_new_cf_0.5 --cf 0.5 --gnn gcn --megu --cacdc --gnndelete --scrub

python eval_script.py --gnn gcn --dataset Computers --data_dir /scratch/akshit.sinha/data --attack_type label --db_name rebuttal_test_new_cf_0.5 --cf 0.5 --log_name rebuttal_cf_0.5 --start_seed 0 --end_seed 5 --megu --cacdc --gnndelete --scrub

#######################

python run_hp_tune.py --dataset DBLP --data_dir /scratch/akshit.sinha/data --df_size 0.5 --random_seed 0 --attack_type label --db_name rebuttal_test_new_cf_0.5 --cf 0.5 --gnn gcn --megu --cacdc --gnndelete --scrub

python eval_script.py --gnn gcn --dataset DBLP --data_dir /scratch/akshit.sinha/data --attack_type label --db_name rebuttal_test_new_cf_0.5 --cf 0.5 --log_name rebuttal_cf_0.5 --start_seed 0 --end_seed 5 --megu --cacdc --gnndelete --scrub

#######################

#######################
#######################

python run_hp_tune.py --dataset Physics --data_dir /scratch/akshit.sinha/data --df_size 0.5 --random_seed 0 --attack_type label --db_name rebuttal_test_new_cf_0.25 --cf 0.25 --gnn gcn --megu --cacdc --gnndelete --scrub

python eval_script.py --gnn gcn --dataset Physics --data_dir /scratch/akshit.sinha/data --attack_type label --db_name rebuttal_test_new_cf_0.25 --cf 0.25 --log_name rebuttal_cf_0.25 --start_seed 0 --end_seed 5 --megu --cacdc --gnndelete --scrub

#######################

python run_hp_tune.py --dataset Computers --data_dir /scratch/akshit.sinha/data --df_size 0.5 --random_seed 0 --attack_type label --db_name rebuttal_test_new_cf_0.25 --cf 0.25 --gnn gcn --megu --cacdc --gnndelete --scrub

python eval_script.py --gnn gcn --dataset Computers --data_dir /scratch/akshit.sinha/data --attack_type label --db_name rebuttal_test_new_cf_0.25 --cf 0.25 --log_name rebuttal_cf_0.25 --start_seed 0 --end_seed 5 --megu --cacdc --gnndelete --scrub

#######################

python run_hp_tune.py --dataset DBLP --data_dir /scratch/akshit.sinha/data --df_size 0.5 --random_seed 0 --attack_type label --db_name rebuttal_test_new_cf_0.25 --cf 0.25 --gnn gcn --megu --cacdc --gnndelete --scrub

python eval_script.py --gnn gcn --dataset DBLP --data_dir /scratch/akshit.sinha/data --attack_type label --db_name rebuttal_test_new_cf_0.25 --cf 0.25 --log_name rebuttal_cf_0.25 --start_seed 0 --end_seed 5 --megu --cacdc --gnndelete --scrub

#######################

#######################
#######################

python run_hp_tune.py --dataset Physics --data_dir /scratch/akshit.sinha/data --df_size 0.5 --random_seed 0 --attack_type label --db_name rebuttal_test_new_cf_0.75 --cf 0.75 --gnn gcn --megu --cacdc --gnndelete --scrub

python eval_script.py --gnn gcn --dataset Physics --data_dir /scratch/akshit.sinha/data --attack_type label --db_name rebuttal_test_new_cf_0.75 --cf 0.75 --log_name rebuttal_cf_0.75 --start_seed 0 --end_seed 5 --megu --cacdc --gnndelete --scrub

#######################

python run_hp_tune.py --dataset Computers --data_dir /scratch/akshit.sinha/data --df_size 0.5 --random_seed 0 --attack_type label --db_name rebuttal_test_new_cf_0.75 --cf 0.75 --gnn gcn --megu --cacdc --gnndelete --scrub

python eval_script.py --gnn gcn --dataset Computers --data_dir /scratch/akshit.sinha/data --attack_type label --db_name rebuttal_test_new_cf_0.75 --cf 0.75 --log_name rebuttal_cf_0.75 --start_seed 0 --end_seed 5 --megu --cacdc --gnndelete --scrub

#######################

python run_hp_tune.py --dataset DBLP --data_dir /scratch/akshit.sinha/data --df_size 0.5 --random_seed 0 --attack_type label --db_name rebuttal_test_new_cf_0.75 --cf 0.75 --gnn gcn --megu --cacdc --gnndelete --scrub

python eval_script.py --gnn gcn --dataset DBLP --data_dir /scratch/akshit.sinha/data --attack_type label --db_name rebuttal_test_new_cf_0.75 --cf 0.75 --log_name rebuttal_cf_0.75 --start_seed 0 --end_seed 5 --megu --cacdc --gnndelete --scrub

#######################