# load all files of the form indices_*.pt

import os

def load_indices(path):
    files = os.listdir(path)
    indices = []
    for file in files:
        if file.startswith('indices_') and file.endswith('.pt'):
            # extract the index from the filename
            index = int(file[8:-3])
            indices.append(index+1)
    indices.sort(reverse=True)
    return indices

def run_cmd(epoch):
    cmd = f" python main.py --dataset Cora --data_dir /scratch/akshit.sinha/data --db_name exp_ks_2 --unlearning_model cacdc --training_epochs {epoch} --random_seed 0"
    
    print(cmd)
    
    os.system(cmd)
    

epoch_idxs = load_indices('.')
print(epoch_idxs)

for idx in epoch_idxs:
    run_cmd(idx)
    print(f"Done with {idx}")
    break