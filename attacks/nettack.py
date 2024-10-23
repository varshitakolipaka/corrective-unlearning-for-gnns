import torch
import numpy as np
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.defense import GCN
from torch_geometric.utils import to_scipy_sparse_matrix

def nettack_attack(data, epsilon, seed, target_node=None, num_budgets=None):
    """
    Performs Nettack attack on a target node.
    
    :param data: PyG data object
    :param epsilon: Fraction of perturbations (equivalent to num_budgets)
    :param seed: Random seed
    :param target_node: Node to attack (if None, selects a random node)
    :param num_budgets: Number of perturbations allowed (can be derived from epsilon)
    :return: data with poisoned features and adjacency
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    data = data.cpu()
    adj_matrix = to_scipy_sparse_matrix(data.edge_index).tocsc()  # Sparse adjacency matrix
    features = data.x.numpy()  # Node features
    labels = data.y.numpy()  # Labels
    
    if target_node is None:
        target_node = np.random.choice(np.arange(data.num_nodes))  # Pick random node if not specified
    
    if num_budgets is None:
        num_budgets = int(epsilon * adj_matrix.shape[0])  # Define perturbation budget from epsilon

    # Setup surrogate GCN model (You may need to adjust GCN parameters)
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    idx_train = data.train_mask.nonzero(as_tuple=False).view(-1)
    surrogate.fit(features, adj_matrix, labels, idx_train, idx_val=None, patience=30)
    
    # Setup Nettack
    nettack = Nettack(surrogate, nnodes=adj_matrix.shape[0], attack_structure=True, attack_features=True, device='cpu').to('cpu')
    
    # Attack the target node
    nettack.attack(features, adj_matrix, labels, target_node, n_perturbations=num_budgets)
    
    # Apply perturbations to the graph
    perturbed_adj = nettack.modified_adj
    perturbed_features = nettack.modified_features

    # Update the data object
    data.edge_index = torch.tensor(perturbed_adj.nonzero(), dtype=torch.long)
    data.x = torch.tensor(perturbed_features, dtype=torch.float)
    
    # Log poison information
    data.poison_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.poison_mask[target_node] = True
    data.poisoned_nodes = torch.tensor([target_node])
    
    print(f"Performed Nettack on node {target_node} with {num_budgets} perturbations.")
    
    return data, target_node
