import os

fracs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for frac in fracs:
    command = f"python main.py --unlearning_model cacdc --dataset Cora --attack_type label --db_name label_main --k_frac {frac}"
    os.system(command)