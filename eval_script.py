import os

def get_script(dataset, unlearning_model, attack, seed):
    
    dataset_to_df = {
        'Amazon': 10000,
        'Cora': 750,
        'CS': 3000,
    }
    
    if attack == 'label':
        return f"python main.py --df_size 0.5 --dataset {dataset} --unlearning_model {unlearning_model} --attack_type label --random_seed {seed} --gnn gcn  --data_dir /scratch/akshit.sinha/data"

    if attack == 'edge':
        return f"python main.py --df_size {dataset_to_df[dataset]} --dataset {dataset} --unlearning_model {unlearning_model} --attack_type edge --request edge --random_seed {seed} --data_dir /scratch/akshit.sinha/data"

unlearning_models = ['utu', 'scrub','gnndelete','megu','gif','contra_2', 'retrain', 'grub']
# unlearning_models = ['utu', 'gnndelete', 'megu', 'gif', 'retrain']
# unlearning_models = ['contra_2']
# attacks = ['label', 'edge']
attacks = ['edge']
datasets = ['CS', 'Cora', 'Amazon']
# datasets = ['Cora']

for dataset in datasets:
    for seed in range(10):
        for unlearning_model in unlearning_models:
            for attack in attacks:
                    script = get_script(dataset, unlearning_model, attack, seed)
                    # print(script)
                    os.system(script)