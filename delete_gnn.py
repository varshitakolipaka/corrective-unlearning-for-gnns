import os
import copy
import json
from framework.trainer.label_poison import get_label_poisoned_data
from framework.trainer.edge_poison import get_edge_poisoned_data
# import wandb
import pickle
import argparse
import torch
import torch.nn as nn
from torch_geometric.utils import to_undirected, to_networkx, k_hop_subgraph, is_undirected
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.seed import seed_everything
from torch_geometric.utils import subgraph
import copy

from framework import get_model, get_trainer
from framework.training_args import parse_args
from framework.utils import *
from framework.data_loader import split_forget_retain, train_test_split_edges_no_neg_adj_mask, get_original_data
# from train_mi import load_mi_models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
forget="edge"

def get_processed_data(args, val_ratio, test_ratio, df_ratio, subset='in'):
    '''pend for future use'''
    data = get_original_data(args.dataset)
    if args.request == 'edge':
        data = train_test_split_edges_no_neg_adj_mask(data, val_ratio, test_ratio)
        data = split_forget_retain(data, df_ratio, subset)
    elif forget=="node":
        data, flipped_indices = get_label_poisoned_data(args, data, df_ratio, args.random_seed)
        # need to define df_mask and dr_mask
        # once those are done we can also define sdf_mask for gnndelete to work
        # size of df mask should be number of edges in data.train_mask

        # Create a subgraph containing only the nodes in the train mask

        # if we are doing label flip
        # train_nodes = torch.where(data.train_mask)[0]
        # train_edges, _ = subgraph(train_nodes, data.edge_index, relabel_nodes=True)

        data.df_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
        data.dr_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)

        for node in flipped_indices:
            data.train_mask[node] = False
            # df mask should be all the edges connected to node
            node_tensor = torch.tensor([node], dtype=torch.long)
            _, local_edges, _, mask = k_hop_subgraph(
                node_tensor, 1, data.edge_index, num_nodes=data.num_nodes)

            # print("-----------")
            # print(local_edges)

            data.df_mask[mask] = True

        data.dr_mask = ~data.df_mask
        # we just create sdf masks also

    elif forget=="edge":
        #Return type: new edge index, to_indices, from_indices
        augmented_edges, poisoned_indices = get_edge_poisoned_data(args, data, df_ratio, args.random_seed)
        data.edge_index= augmented_edges
        data.df_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
        data.dr_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)

        data.df_mask[poisoned_indices] = True
        data.dr_mask = ~data.df_mask

    return data

torch.autograd.set_detect_anomaly(True)
def main():
    args = parse_args()
    original_path = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'lf_attack', 'in-' + str(args.df_size) + '-' + str(args.random_seed))
    attack_path_all = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'member_infer_all', str(args.random_seed))
    args.attack_dir = attack_path_all
    if not os.path.exists(attack_path_all):
        os.makedirs(attack_path_all, exist_ok=True)
    shadow_path_all = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'shadow_all', str(args.random_seed))
    args.shadow_dir = shadow_path_all
    if not os.path.exists(shadow_path_all):
        os.makedirs(shadow_path_all, exist_ok=True)

    args.checkpoint_dir = os.path.join(
        args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model,
        '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    seed_everything(args.random_seed)

    # Dataset
    data = get_processed_data(args, val_ratio=0.05, test_ratio=0.05, df_ratio=args.df_size)
    print('Directed dataset:', data)

    if args.gnn not in ['rgcn', 'rgat']:
        args.in_dim = data.x.shape[1]

    print('Training args', args)

    # Model
    # model = get_model(args, data.sdf_node_1hop_mask, data.sdf_node_2hop_mask, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type) # this is for gnndelete
    model = get_model(args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)

    if args.unlearning_model != 'retrain':  # Start from trained GNN model
        if os.path.exists(os.path.join(original_path, 'pred_proba.pt')):
            logits_ori = torch.load(os.path.join(original_path, 'pred_proba.pt'))   # logits_ori: tensor.shape([num_nodes, num_nodes]), represent probability of edge existence between any two nodes
            if logits_ori is not None:
                logits_ori = logits_ori.to(device)
        else:
            logits_ori = None

        model_ckpt = torch.load(os.path.join(original_path, 'model_best.pt'), map_location=device)
        model.load_state_dict(model_ckpt['model_state'], strict=False)

    else:       # Initialize a new GNN model
        retrain = None
        logits_ori = None

    model = model.to(device)
    # data = data.to(device)

    # Optimizer
    if 'gnndelete' in args.unlearning_model:
        parameters_to_optimize = [
            {'params': [p for n, p in model.named_parameters() if 'del' in n], 'weight_decay': args.weight_decay}
        ]
        print('parameters_to_optimize', [n for n, p in model.named_parameters() if 'del' in n])
        if 'layerwise' in args.loss_type:
            optimizer1 = torch.optim.Adam(model.deletion1.parameters(), lr=args.lr)
            optimizer2 = torch.optim.Adam(model.deletion2.parameters(), lr=args.lr)
            optimizer = [optimizer1, optimizer2]
        else:
            optimizer = torch.optim.Adam(parameters_to_optimize, lr=args.lr)
    else:
        parameters_to_optimize = [
            {'params': [p for n, p in model.named_parameters()], 'weight_decay': args.weight_decay}
        ]
        print('parameters_to_optimize', [n for n, p in model.named_parameters()])
        optimizer = torch.optim.Adam(parameters_to_optimize, lr=args.lr)

    # wandb.init(config=args, project="GNNDelete", group="over_unlearn", name=get_run_name(args), mode=args.mode)
    # wandb.watch(model, log_freq=100)

    # tqdm.tqdm.write('Hola')

    # MI attack model
    attack_model_all = None
    attack_model_sub = None

    # Train
    trainer = get_trainer(args)
    print('Trainer: ', trainer)

    print(f"df mask: {data.df_mask.sum().item()}") # 5702
    print(f"dr mask: {data.dr_mask.sum().item()}") # 108452 -> are these edges?
    print(data.df_mask.sum().item() + data.dr_mask.sum().item() == data.edge_index.size(1)) # True
    print(f"length of data.x: {data.x.size(dim=0)}") # The length of the x is 19763.

    unlearnt_model = copy.deepcopy(model)
    trainer.train(unlearnt_model, data, optimizer, args)

    # Test
    if args.unlearning_model != 'retrain':
        retrain_path = os.path.join(
            'checkpoint', args.dataset, args.gnn, 'retrain',
            '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]),
            'model_best.pt')
        if os.path.exists(retrain_path):
            retrain_ckpt = torch.load(retrain_path, map_location=device)
            retrain_args = copy.deepcopy(args)
            retrain_args.unlearning_model = 'retrain'
            retrain = get_model(retrain_args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)
            retrain.load_state_dict(retrain_ckpt['model_state'])
            retrain = retrain.to(device)
            retrain.eval()
        else:
            retrain = None
    else:
        retrain = None

    test_results = trainer.test(unlearnt_model, data, model_retrain=None, attack_model_all=attack_model_all, attack_model_sub=attack_model_sub, is_dr = True)
    print(test_results[-1])

    trainer.save_log()
    # wandb.finish()


if __name__ == "__main__":
    main()
