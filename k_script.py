import os

fracs=[0.068, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.75, 1]
for frac in fracs:
    command = f"python main.py --unlearning_model cacdc --random_seed 1 --dataset Cora --attack_type label --db_name rebuttal --k_frac {frac}"
    os.system(command)