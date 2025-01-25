from collections import defaultdict
import copy
import json
import os
import torch
from framework import utils
from framework.training_args import parse_args
from models.deletion import GCNDelete
from models.models import GCN
from trainers.base import Trainer
from attacks.edge_attack import edge_attack_specific_nodes
from attacks.label_flip import label_flip_attack, label_flip_attack_strong
from attacks.feature_attack import trigger_attack
import optuna
from optuna.samplers import TPESampler
from functools import partial
from logger import Logger
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

args = parse_args()

utils.seed_everything(args.random_seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"Using device: {device}")

with open("classes_to_poison.json", "r") as f:
    class_dataset_dict = json.load(f)

with open("model_seeds.json") as f:
    model_seeds = json.load(f)
    # convert to defaultdict
    model_seeds = defaultdict(lambda: 0, model_seeds)    

def train():    # dataset
    
    
    print("==TRAINING==")
    clean_data = utils.get_original_data(args.dataset)
    # utils.train_test_split(
    #     clean_data, args.random_seed, args.train_ratio, args.val_ratio
    # )
    utils.train_test_split(
        clean_data, model_seeds[args.dataset], args.train_ratio, args.val_ratio
    )
    utils.prints_stats(clean_data)
    clean_model = utils.get_model(
        args, clean_data.num_features, args.hidden_dim, clean_data.num_classes
    )
    optimizer = torch.optim.Adam(
        clean_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay
    )
    clean_trainer = Trainer(clean_model, clean_data, optimizer, args)
    
    if args.train_oracle:
        clean_trainer.train()
    else:
        print("Not training oracle")
        return clean_data

    if args.attack_type != "trigger":
        print("ACC__ : ", clean_trainer.evaluate())
        forg, util, forget_f1, util_f1 = clean_trainer.get_score(
            args.attack_type,
            class1=class_dataset_dict[args.dataset]["class1"],
            class2=class_dataset_dict[args.dataset]["class2"],
        )

        print(
            f"==OG Model==\nForg Accuracy: {forg}, Util Accuracy: {util}, Forg F1: {forget_f1}, Util F1: {util_f1}"
        )
        
    # save the clean model
    os.makedirs(args.data_dir, exist_ok=True)
    torch.save(
        clean_model,
        f"{args.data_dir}/{args.gnn}_{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_clean_model.pt",
    )
    return clean_data

def poison(clean_data=None):
    if clean_data is None:
        # load the poisoned data and model and indices from np file
        poisoned_data = torch.load(
            f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_poisoned_data.pt"
        )
        poisoned_model = torch.load(
            f"{args.data_dir}/{args.gnn}_{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_poisoned_model.pt"
        )

        print(poisoned_model.state_dict().keys())

        if args.attack_type == "edge":
            poisoned_indices = poisoned_data.poisoned_edge_indices
        else:
            poisoned_indices = poisoned_data.poisoned_nodes

        optimizer = torch.optim.Adam(
            poisoned_model.parameters(),
            lr=args.train_lr,
            weight_decay=args.weight_decay,
        )
        poisoned_trainer = Trainer(poisoned_model, poisoned_data, optimizer, args)
        poisoned_trainer.evaluate()
        

        forg, util, forget_f1, util_f1 = poisoned_trainer.get_score(
            args.attack_type,
            class1=class_dataset_dict[args.dataset]["class1"],
            class2=class_dataset_dict[args.dataset]["class2"],
        )

        print(
            f"==Poisoned Model==\nForg Accuracy: {forg}, Util Accuracy: {util}, Forg F1: {forget_f1}, Util F1: {util_f1}"
        )
        # print(poisoned_trainer.calculate_PSR())
        return poisoned_data, poisoned_indices, poisoned_model

    print("==POISONING==")
    if args.attack_type == "label":
        poisoned_data, poisoned_indices = label_flip_attack(
            clean_data,
            args.df_size,
            args.random_seed,
            class_dataset_dict[args.dataset]["class1"],
            class_dataset_dict[args.dataset]["class2"],
        )
    if args.attack_type == "label_strong":
        poisoned_data, poisoned_indices = label_flip_attack_strong(
            clean_data,
            args.df_size,
            args.random_seed,
            class_dataset_dict[args.dataset]["class1"],
            class_dataset_dict[args.dataset]["class2"],
        )
    elif args.attack_type == "edge":
        poisoned_data, poisoned_indices = edge_attack_specific_nodes(
            clean_data,
            args.df_size,
            args.random_seed,
            class1=class_dataset_dict[args.dataset]["class1"],
            class2=class_dataset_dict[args.dataset]["class2"],
        )
    elif args.attack_type == "random":
        poisoned_data = copy.deepcopy(clean_data)
        poisoned_indices = torch.randperm(clean_data.num_nodes)[
            : int(clean_data.num_nodes * args.df_size)
        ]
        poisoned_data.poisoned_nodes = poisoned_indices
    elif args.attack_type == "trigger":
        poisoned_data, poisoned_indices = trigger_attack(
            clean_data,
            args.df_size,
            args.random_seed,
            victim_class=class_dataset_dict[args.dataset]["victim_class"],
            target_class=class_dataset_dict[args.dataset]["target_class"],
            trigger_size=args.trigger_size,
        )
    poisoned_data = poisoned_data.to(device)

    poisoned_model = utils.get_model(
        args, poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes
    )

    optimizer = torch.optim.Adam(
        poisoned_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay
    )
    poisoned_trainer = Trainer(poisoned_model, poisoned_data, optimizer, args)
    poisoned_trainer.train()

    # save the poisoned data and model and indices to np file
    forg, util, forget_f1, util_f1 = poisoned_trainer.get_score(
        args.attack_type,
        class1=class_dataset_dict[args.dataset]["class1"],
        class2=class_dataset_dict[args.dataset]["class2"],
    )

    print(
        f"==Poisoned Model==\nForg Accuracy: {forg}, Util Accuracy: {util}, Forg F1: {forget_f1}, Util F1: {util_f1}"
    )
    
    # ask the user if they want to save the model
    save_model = input("Do you want to save the model? (y/n): ")
    
    if save_model == "y":

        os.makedirs(args.data_dir, exist_ok=True)

        torch.save(
            poisoned_model,
            f"{args.data_dir}/{args.gnn}_{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_poisoned_model.pt",
        )

        torch.save(
            poisoned_data,
            f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_poisoned_data.pt",
        )
    else:
        print("Model not saved")


    # print(f"PSR: {poisoned_trainer.calculate_PSR()}")
    return poisoned_data, poisoned_indices, poisoned_model

if __name__ == "__main__":
    print("\n\n\n")
    print(args.dataset, args.attack_type)
    
    # set best gnn params
    with open ("base_params.json", "r") as f:
        data = json.load(f)
        if args.dataset in data:
            if args.gnn in data[args.dataset]:
                best_params = data[args.dataset][args.gnn]
                for key, value in best_params.items():
                    setattr(args, key, value)
        
    clean_data = train()
    poisoned_data, poisoned_indices, poisoned_model = poison(clean_data)