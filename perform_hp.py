import os

def get_script(dataset, unlearning_model, attack, seed=None):
    if seed == None:
        seed = 1
        
    dataset_to_df = {
        'Amazon': 30000,
        'Cora': 5000,
        'PubMed': 25000,
    }
    
    if attack == 'label':
        return f"python hp_tune.py --df_size 0.5 --dataset {dataset} --unlearning_model {unlearning_model} --attack_type label --random_seed {seed} --data_dir /scratch/akshit.sinha/data"

    if attack == 'edge':
        return f"python hp_tune.py --df_size {dataset_to_df[dataset]} --dataset {dataset} --unlearning_model {unlearning_model} --attack_type edge --request edge --random_seed {seed} --data_dir /scratch/akshit.sinha/data"

unlearning_models = ['utu', 'scrub','gnndelete','megu','gif','contrastive', 'retrain']
# unlearning_models = ['contrastive']
# attacks = ['label', 'edge']
attacks = ['label']
datasets = ['Amazon', 'Cora', 'PubMed', 'CS']

for dataset in datasets:
    for unlearning_model in unlearning_models:
        for attack in attacks:
            script = get_script(dataset, unlearning_model, attack)
            print(script)
            os.system(script)