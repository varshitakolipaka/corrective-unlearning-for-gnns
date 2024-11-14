import os

fracs=[0.001, 0.002, 0.008, 0.01, 0.015, 0.020, 0.025, 0.030]
for count in range(0, 5):
    for frac in fracs:
        command = f"python main.py --unlearning_model cacdc --random_seed 1 --dataset Cora --attack_type label --db_name label_main --k_frac {frac} --counter {count}"
        os.system(command)