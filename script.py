import os

#this script is plotting the performance of forg and util vs the frac
#call this setting test frac test

fracs = [0.068, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.75, 1]
for frac in fracs:
    edge_command = f"python main.py --k_frac {frac} --random_seed 1 --frac_test_filename ./report_edge.txt --frac_test True --unlearning_model cacdc --db_name rebuttal --attack_type edge --request edge --df_size 750 --dataset Cora"
    label_command = f"python main.py --k_frac {frac} --random_seed 1 --frac_test True --unlearning_model cacdc --db_name rebuttal --attack_type label"
    os.system(edge_command)
    os.system(label_command)