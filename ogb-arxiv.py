import argparse
import copy
from functools import partial
import json
import os
import torch
import torch.nn.functional as F
import numpy as np

import torch_geometric.transforms as T
from framework import utils
from framework.training_args import parse_args
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, f1_score
import optuna
from optuna.samplers import TPESampler
# from models import GCN

args = parse_args()
utils.seed_everything(args.random_seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

with open("classes_to_poison.json", "r") as f:
    class_dataset_dict = json.load(f)


with open("model_seeds.json") as f:
    model_seeds = json.load(f)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

def label_flip_attack(data, split_idx, epsilon, seed, class1=None, class2=None):
    np.random.seed(seed)
    
    # Move data to CPU for easier manipulation
    data = data.cpu()
    
    # Get training indices from OGB split index
    train_indices = split_idx['train']
    labels = data.y.squeeze()
    train_labels, counts = torch.unique(labels[train_indices], return_counts=True)

    # Print class distribution in the training set
    for i, label in enumerate(train_labels):
        print(f"Class {label.item()}: {counts[i].item()}")

    # Get indices for nodes belonging to class1 and class2
    class1_indices = train_indices[labels[train_indices] == class1]
    class2_indices = train_indices[labels[train_indices] == class2]

    # Determine the number of labels to flip, based on epsilon and class sizes
    epsilon = min(epsilon, 0.5)
    num_flips = int(epsilon * min(len(class1_indices), len(class2_indices)))

    print(f"Flipping {num_flips} labels from class {class1} to class {class2} and vice versa")

    # Randomly select indices to flip
    flip_indices_class1 = np.random.choice(class1_indices.numpy(), num_flips, replace=False)
    flip_indices_class2 = np.random.choice(class2_indices.numpy(), num_flips, replace=False)

    # Apply label flips
    data.y[flip_indices_class1] = class2
    data.y[flip_indices_class2] = class1

    # Metadata for tracking the poisoned data
    data.class1 = class1
    data.class2 = class2
    flipped_indices = np.concatenate([flip_indices_class1, flip_indices_class2])
    data.poisoned_nodes = torch.tensor(flipped_indices)

    # Create a mask for poisoned nodes
    data.poison_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.poison_mask[flipped_indices] = True

    print(f"Poisoned {num_flips} labels in total from class {class1} and class {class2}")
    
    return data, flipped_indices

def subset_acc(data, pred, class1, class2, num_classes):
    poisoned_classes = [class1, class2]
    true_labels = data.y.to(device)
    pred_labels = pred.to(device)

    clean_classes = [i for i in range(num_classes) if i not in poisoned_classes]
    
    acc_poisoned = np.mean([accuracy_score(true_labels[true_labels == c].cpu(), 
                                           pred_labels[true_labels == c].cpu()) 
                            for c in poisoned_classes if (true_labels == c).sum() > 0])
    
    acc_clean = np.mean([accuracy_score(true_labels[true_labels == c].cpu(), 
                                        pred_labels[true_labels == c].cpu()) 
                         for c in clean_classes if (true_labels == c).sum() > 0])

    f1_poisoned = np.mean([f1_score(true_labels[true_labels == c].cpu(), 
                                    pred_labels[true_labels == c].cpu(), average="weighted") 
                           for c in poisoned_classes if (true_labels == c).sum() > 0])
    
    f1_clean = np.mean([f1_score(true_labels[true_labels == c].cpu(), 
                                 pred_labels[true_labels == c].cpu(), average="weighted") 
                        for c in clean_classes if (true_labels == c).sum() > 0])

    return acc_poisoned, acc_clean, f1_poisoned, f1_clean

@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()
    out = F.log_softmax(model(data.x, data.adj_t), dim=-1)
    y_pred = out.argmax(dim=-1, keepdim=True)
    
    train_acc = evaluator.eval({'y_true': data.y[split_idx['train']], 'y_pred': y_pred[split_idx['train']]})['acc']
    valid_acc = evaluator.eval({'y_true': data.y[split_idx['valid']], 'y_pred': y_pred[split_idx['valid']]})['acc']
    test_acc = evaluator.eval({'y_true': data.y[split_idx['test']], 'y_pred': y_pred[split_idx['test']]})['acc']

    return train_acc, valid_acc, test_acc, y_pred

def train(load=False):
        
    print("====TRAINING====")
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    utils.prints_stats_ogb(data, split_idx, dataset.num_classes)
    
    if load:
        model = torch.load(
            f"{args.data_dir}/{args.gnn}_{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_clean_model.pt"
        )
        model.eval()
        train_acc, valid_acc, test_acc, y_pred = test(model, data, split_idx, evaluator)
        forg, util, forget_f1, util_f1 = subset_acc(data, y_pred, class1=4, class2=26, num_classes=dataset.num_classes)
        
        print(f'Train Acc: {train_acc:.4f}, Val Acc: {valid_acc:.4f}, Test Acc: {test_acc:.4f}')
        print(
            f"Forg Accuracy: {forg}, Util Accuracy: {util}, Forg F1: {forget_f1}, Util F1: {util_f1}"
        )
        return data, split_idx, dataset.num_classes

    model = GCN(data.num_features, args.hidden_dim, dataset.num_classes, num_layers=3, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.adj_t)[train_idx]
        loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()

        # Evaluate accuracy on train, validation, and test splits
        train_acc, valid_acc, test_acc, y_pred = test(model, data, split_idx, evaluator)
        forg, util, forget_f1, util_f1 = subset_acc(data, y_pred, class1=4, class2=26, num_classes=dataset.num_classes)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {valid_acc:.4f}, Test Acc: {test_acc:.4f}')
            print(
            f"Forg Accuracy: {forg}, Util Accuracy: {util}, Forg F1: {forget_f1}, Util F1: {util_f1}"
            )
    
    os.makedirs(args.data_dir, exist_ok=True)
    torch.save(
        model,
        f"{args.data_dir}/{args.gnn}_{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_clean_model.pt",
    )
            
    return data, split_idx, dataset.num_classes

def poison(clean_data, split_idx, num_classes, load=False):
    
    evaluator = Evaluator(name='ogbn-arxiv')
    
    if load:
        poisoned_data = torch.load(
            f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_poisoned_data.pt"
        )
        poisoned_model = torch.load(
            f"{args.data_dir}/{args.gnn}_{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_poisoned_model.pt"
        )
        poisoned_model.eval()
        train_acc, valid_acc, test_acc, y_pred = test(poisoned_model, poisoned_data, split_idx, evaluator)
        forg, util, forget_f1, util_f1 = subset_acc(poisoned_data, y_pred, class1=4, class2=26, num_classes=num_classes)
        
        print(f'Train Acc: {train_acc:.4f}, Val Acc: {valid_acc:.4f}, Test Acc: {test_acc:.4f}')
        print(
            f"Forg Accuracy: {forg}, Util Accuracy: {util}, Forg F1: {forget_f1}, Util F1: {util_f1}"
        )
        return poisoned_data, poisoned_data.poisoned_nodes, poisoned_model
    
    
    print("====POISONING====")
    poisoned_data, poisoned_indices = label_flip_attack(
        clean_data,
        split_idx,
        args.df_size,
        args.random_seed,
        class1= 4,
        class2= 26
    )
    poisoned_data = poisoned_data.to(device)
    train_idx = split_idx['train'].to(device)

    poisoned_model = GCN(poisoned_data.num_features, args.hidden_dim, num_classes, num_layers=3, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(poisoned_model.parameters(), lr=0.01)

    for epoch in range(1, 301):
        poisoned_model.train()
        optimizer.zero_grad()
        out = poisoned_model(poisoned_data.x, poisoned_data.adj_t)[train_idx]
        loss = F.nll_loss(out, poisoned_data.y.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()

        # Evaluate accuracy on train, validation, and test splits
        train_acc, valid_acc, test_acc, y_pred = test(poisoned_model, poisoned_data, split_idx, evaluator)
        forg, util, forget_f1, util_f1 = subset_acc(poisoned_data, y_pred, class1=4, class2=26, num_classes=num_classes)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {valid_acc:.4f}, Test Acc: {test_acc:.4f}')
            print(
            f"Forg Accuracy: {forg}, Util Accuracy: {util}, Forg F1: {forget_f1}, Util F1: {util_f1}"
            )
        
        
    torch.save(
        poisoned_model,
        f"{args.data_dir}/{args.gnn}_{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_poisoned_model.pt",
    )

    torch.save(
        poisoned_data,
        f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_poisoned_data.pt",
    )

    return poisoned_data, poisoned_indices, poisoned_model

def unlearn(split_idx, poisoned_data, poisoned_indices, poisoned_model):
    print("==UNLEARNING==")
    print("waiting for sparse matrix to convert into edge index ...")
    utils.find_masks(
        poisoned_data, poisoned_indices, split_idx, args, attack_type=args.attack_type
    )
    # print the sum of dr and df masks of poisoned data
    print(f"Sum of dr mask: {poisoned_data.dr_mask.sum()}")
    print(f"Sum of df mask: {poisoned_data.df_mask.sum()}")

hp_tuning_params_dict = {
    "retrain": {
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-5, 1e-1, "log"),
        "unlearning_epochs": (600, 1400, "int"),
    },
    "gnndelete": {
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-5, 1e-1, "log"),
        # "unlearning_epochs": (10, 200, "int"),
        "alpha": (0, 1, "float"),
        "loss_type": (
            [
                "both_all",
                "both_layerwise",
                "only2_layerwise",
                "only2_all",
                "only1",
                "only3",
                "only3_all",
                "only3_layerwise",
            ],
            "categorical",
        ),
    },
    "gnndelete_ni": {
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-5, 1e-1, "log"),
        "unlearning_epochs": (10, 100, "int"),
        "loss_type": (
            [
                "only2_layerwise",
                "only2_all",
                "only1",
                "only3",
                "only3_all",
                "only3_layerwise",
            ],
            "categorical",
        ),
    },
    "gif": {
        "iteration": (10, 1000, "int"),
        "scale": (1e7, 1e11, "log"),
        "damp": (0.0, 1.0, "float"),
    },
    "gradient_ascent": {
        # "unlearning_epochs": (10, 2000, "int"),
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-5, 1e-1, "log"),
    },
    "contrastive": {
        "contrastive_epochs_1": (5, 30, "int"),
        "contrastive_epochs_2": (5, 30, "int"),
        # "maximise_epochs": (5, 30, "int"),
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-5, 1e-1, "log"),
        "contrastive_margin": (1, 1e3, "log"),
        "contrastive_lambda": (0.0, 1.0, "float"),
        "contrastive_frac": (0.01, 0.2, "float"),
        "k_hop": (1, 2, "int"),
    },
    "contra_2": {
        "contrastive_epochs_1": (1, 10, "int"),
        "contrastive_epochs_2": (1, 30, "int"),
        "steps": (1, 10, "int"),
        # "maximise_epochs": (5, 30, "int"),
        "unlearn_lr": (1e-4, 1e-1, "log"),
        # "contrastive_margin": (1, 10, "log"),
        # "contrastive_lambda": (0.0, 1.0, "float"),
        "contrastive_frac": (0.02, 0.3, "float"),
        "k_hop": (1, 3, "int"),
        # "ascent_lr": (1e-6, 1e-3, "log"),
        "descent_lr": (1e-4, 1e-1, "log"),
        # "scrubAlpha": (1e-6, 10, "log"),
    },
    "contrascent": {
        "contrastive_epochs_1": (1, 5, "int"),
        "contrastive_epochs_2": (1, 5, "int"),
        "steps": (1, 15, "int"),
        # "maximise_epochs": (5, 30, "int"),
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "contrastive_margin": (1, 10, "log"),
        # "contrastive_lambda": (0.0, 1.0, "float"),
        "contrastive_frac": (0.01, 0.2, "float"),
        "k_hop": (1, 3, "int"),
        "ascent_lr": (1e-5, 1e-3, "log"),
        "descent_lr": (1e-5, 1e-1, "log"),
        "scrubAlpha": (1e-6, 10, "log"),
    },
    "cacdc": {
        "contrastive_epochs_1": (1, 6, "int"),
        "contrastive_epochs_2": (1, 50, "int"),
        "steps": (1, 2, "int"),
        # "maximise_epochs": (5, 30, "int"),
        "unlearn_lr": (1e-4, 1e-1, "log"),
        # "contrastive_margin": (1, 10, "log"),
        # "contrastive_lambda": (0.0, 1.0, "float"),
        "contrastive_frac": (0.02, 0.3, "float"),
        "k_hop": (1, 3, "int"),
        "ascent_lr": (1e-6, 1e-3, "log"),
        "descent_lr": (1e-4, 1e-1, "log"),
        # "scrubAlpha": (1e-6, 10, "log"),
    },
    "utu": {},
    "scrub": {
        "unlearn_iters": (110, 300, "int"),
        # 'kd_T': (1, 10, "float"),
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "scrubAlpha": (1e-6, 10, "log"),
        "msteps": (10, 150, "int"),
        # 'weight_decay': (1e-5, 1e-1, "log"),
    },
    "scrub_no_kl": {
        "unlearn_iters": (110, 200, "int"),
        # 'kd_T': (1, 10, "float"),
        "unlearn_lr": (1e-5, 1e-1, "log"),
        # "scrubAlpha": (1e-6, 10, "log"),
        "msteps": (10, 100, "int"),
        # 'weight_decay': (1e-5, 1e-1, "log"),
    },
    "scrub_no_kl_combined": {
        "unlearn_iters": (10, 50, "int"),
        # 'kd_T': (1, 10, "float"),
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "scrubAlpha": (1e-6, 10, "log"),
        "msteps": (1, 20, "int"),
        # 'weight_decay': (1e-5, 1e-1, "log"),
    },
    "yaum": {
        "unlearn_iters": (10, 300, "int"),
        # 'kd_T': (1, 10, "float"),
        "ascent_lr": (1e-5, 1e-3, "log"),
        "descent_lr": (1e-5, 1e-1, "log"),
        # "scrubAlpha": (1e-6, 10, "log"),
        # "msteps": (10, 100, "int"),
    },
    "megu": {
        "unlearn_lr": (1e-6, 1e-3, "log"),
        # "unlearning_epochs": (10, 1000, "int"),
        "kappa": (1e-3, 1, "log"),
        "alpha1": (0, 1, "float"),
        "alpha2": (0, 1, "float"),
    },
    "ssd": {
        "SSDdampening": (0.1, 10, "log"),
        "SSDselectwt": (0.1, 100, "log"),
    },
    "clean": {
        "train_lr": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-5, 1e-1, "log"),
        "training_epochs": (500, 3000, "int"),
    },
}


def set_hp_tuning_params(trial):
    hp_tuning_params = hp_tuning_params_dict[args.unlearning_model]
    for hp, values in hp_tuning_params.items():
        if values[1] == "categorical":
            setattr(args, hp, trial.suggest_categorical(hp, values[0]))
        elif values[2] == "int":
            setattr(args, hp, trial.suggest_int(hp, values[0], values[1]))
        elif values[2] == "float":
            setattr(args, hp, trial.suggest_float(hp, values[0], values[1]))
        elif values[2] == "log":
            setattr(args, hp, trial.suggest_float(hp, values[0], values[1], log=True))


def objective(trial, model, data):
    # Define the hyperparameters to tune
    set_hp_tuning_params(trial)

    model_internal = copy.deepcopy(model)

    optimizer = utils.get_optimizer(args, model_internal)
    trainer = utils.get_trainer(args, model_internal, data, optimizer)

    _, _, time_taken = trainer.train()

    if args.linked:
        obj = trainer.validate(is_dr=False)  # REAL
    else:
        obj = trainer.validate(is_dr=True)  # REAL

    forget_acc, util_acc, forget_f1, util_f1 = trainer.get_score(
        args.attack_type,
        class1=class_dataset_dict[args.dataset]["class1"],
        class2=class_dataset_dict[args.dataset]["class2"],
    )
    trial.set_user_attr("forget_acc", forget_acc)
    trial.set_user_attr("util_acc", util_acc)
    trial.set_user_attr("forget_f1", forget_f1)
    trial.set_user_attr("util_f1", util_f1)
    trial.set_user_attr("time_taken", time_taken)

    # We want to minimize misclassification rate and maximize accuracy
    return obj

if __name__ == "__main__":
    print("\n\n\n")
    clean_data, split_idx, num_classes = train(load=True)
    poisoned_data, poisoned_indices, poisoned_model = poison(clean_data, split_idx, num_classes, load=True)
    
    if args.corrective_frac < 1:
        print("==POISONING CORRECTIVE==")
        poisoned_indices = poisoned_data.poisoned_nodes
        print(f"No. of poisoned nodes: {len(poisoned_indices)}")
        poisoned_indices = utils.sample_poison_data(poisoned_indices, args.corrective_frac)
        poisoned_data.poisoned_nodes = poisoned_indices
        print(f"No. of poisoned nodes after corrective: {len(poisoned_indices)}")
        
    print("==UNLEARNING==")
    print("waiting for sparse matrix to convert into edge index ...")
    utils.find_masks(
        poisoned_data, poisoned_indices, split_idx, args, attack_type=args.attack_type
    )
    poisoned_data.num_classes = 40
    # print the sum of dr and df masks of poisoned data
    print(f"Sum of dr mask: {poisoned_data.dr_mask.sum()}")
    print(f"Sum of df mask: {poisoned_data.df_mask.sum()}")


    if "gnndelete" in args.unlearning_model:
        # Create a partial function with additional arguments
        model = utils.get_model(
            args,
            poisoned_data.num_features,
            args.hidden_dim,
            poisoned_data.num_classes,
            mask_1hop=poisoned_data.sdf_node_1hop_mask,
            mask_2hop=poisoned_data.sdf_node_2hop_mask,
            mask_3hop=poisoned_data.sdf_node_3hop_mask,
        )

        # copy the weights from the poisoned model
        state_dict = poisoned_model.state_dict()
        state_dict["deletion1.deletion_weight"] = model.deletion1.deletion_weight
        state_dict["deletion2.deletion_weight"] = model.deletion2.deletion_weight
        state_dict["deletion3.deletion_weight"] = model.deletion3.deletion_weight

        model.load_state_dict(state_dict)
    elif "retrain" in args.unlearning_model:
        model = GCN(poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes, num_layers=3, dropout=0.5).to(device)
    else:
        model = GCN(poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes, num_layers=3, dropout=0.5).to(device)
        model.load_state_dict(poisoned_model.state_dict())
        
    # set train val test masks
    poisoned_data.train_mask = torch.zeros(poisoned_data.num_nodes, dtype=torch.bool)
    poisoned_data.train_mask[split_idx["train"]] = True
    poisoned_data.val_mask = torch.zeros(poisoned_data.num_nodes, dtype=torch.bool)
    poisoned_data.val_mask[split_idx["valid"]] = True
    poisoned_data.test_mask = torch.zeros(poisoned_data.num_nodes, dtype=torch.bool)
    poisoned_data.test_mask[split_idx["test"]] = True
    
    # squeeze the labels
    poisoned_data.y = poisoned_data.y.squeeze()
        
    objective_func = partial(objective, model=model, data=poisoned_data)

    print("==HYPERPARAMETER TUNING==")
    # Create a study with TPE sampler
    study = optuna.create_study(
        sampler=TPESampler(seed=42),
        direction="maximize",
        study_name=f"{args.gnn}_{args.dataset}_{args.attack_type}_{args.df_size}_{args.unlearning_model}_{args.random_seed}_{class_dataset_dict[args.dataset]['class1']}_{class_dataset_dict[args.dataset]['class2']}",
        load_if_exists=True,
        storage=f"sqlite:///hp_tuning/new/{args.db_name}.db",
    )

    print("==OPTIMIZING==")

    # Optimize the objective function

    # reduce trials for utu and contrastive
    if args.unlearning_model == "utu":
        study.optimize(objective_func, n_trials=1)
    elif args.unlearning_model == "retrain":
        study.optimize(objective_func, n_trials=30)
    # elif args.unlearning_model == "contrastive" or args.unlearning_model == "contra_2":
    #     study.optimize(objective_func, n_trials=200)
    else:
        study.optimize(objective_func, n_trials=100)