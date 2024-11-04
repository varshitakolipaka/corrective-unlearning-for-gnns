import torch
import numpy as np
import random
import copy
from torch_geometric import utils
from deeprobust.graph.global_attack import NodeEmbeddingAttack
import scipy.sparse as sp

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def metattack(data, epsilon, seed):
    np.random.seed(seed)
    random.seed(seed)
    data= data.cpu()
    if(epsilon<1):
        epsilon= epsilon*data.num_edges

    epsilon = int(epsilon)
    print(epsilon)

    poisoned_data = copy.deepcopy(data)

    # Convert torch_geometric data to sparse adjacency matrix format
    adj = utils.to_scipy_sparse_matrix(poisoned_data.edge_index).astype(np.float32)
    adj = adj.tocsr()
    model = NodeEmbeddingAttack()
    model.attack(adj, attack_type="add", n_candidates=epsilon, n_perturbations=epsilon)
    adj_poisoned = model.modified_adj

    # Convert the modified adjacency back to edge_index format
    poisoned_edge_index, _ = utils.from_scipy_sparse_matrix(adj_poisoned)

    # Identify the edges that were added to the original graph
    original_edges_set = set(tuple(edge) for edge in data.edge_index.t().tolist())
    poisoned_edges_set = set(tuple(edge) for edge in poisoned_edge_index.t().tolist())
    added_edges = poisoned_edges_set - original_edges_set
    added_edges = torch.tensor(list(added_edges), dtype=torch.long).t()

    # Now we find the indices of the added edges in the poisoned_edge_index
    poisoned_indices = []
    for i in range(poisoned_edge_index.size(1)):  # Loop over edges in poisoned_edge_index
        edge = poisoned_edge_index[:, i]
        if (edge[0].item(), edge[1].item()) in added_edges.t().tolist():
            poisoned_indices.append(i)

    poisoned_indices = torch.tensor(poisoned_indices, dtype=torch.long)

    # Update the poisoned data with the new edge indices
    poisoned_data.edge_index = poisoned_edge_index

    return poisoned_data, poisoned_indices