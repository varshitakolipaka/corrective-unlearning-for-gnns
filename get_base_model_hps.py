import json
import optuna
from optuna.samplers import TPESampler
from framework.training_args import parse_args

with open('classes_to_poison.json', 'r') as f:
    class_dataset_dict = json.load(f)

if __name__=="__main__":
    args = parse_args()
    
    study = optuna.create_study(
            sampler=TPESampler(seed=42),
            direction="maximize",
            study_name=f"{args.gnn}_{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}",
            load_if_exists=True,
            storage=f"sqlite:///hp_tuning/base_gnn/{args.db_name}.db",
        )
    
    trials = study.get_trials()
    
    # remove trials with no value
    trials = [trial for trial in trials if trial.value is not None]
    
    # order trials by value
    trials = sorted(trials, key=lambda x: x.value, reverse=True)

    # print the top 10 trials
    for trial in trials[:10]:
        print(f"Trial: {trial.number}, Value: {trial.value}, Params: {trial.params}")
    
    # get best hyperparameters
    best_trial = trials[0]
    print(f"Best trial: {best_trial.value}")
    
    params = best_trial.params
    print(f"Best trial params: {params}")
    
    # save to file
    try:
        with open('base_params.json', 'r') as f:
            data = json.load(f)
    except:
        data = {}
        
    with open('base_params.json', 'w') as f:
        
        # create the data object if it doesn't exist
        if args.dataset not in data:
            data[args.dataset] = {}
        if args.gnn not in data[args.dataset]:
            data[args.dataset][args.gnn] = {}
            
        # update the data object for the dataset and gnn key
        data[args.dataset][args.gnn] = params
        
        json.dump(data, f, indent=4)