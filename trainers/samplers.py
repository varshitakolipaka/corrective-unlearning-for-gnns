import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
import scipy.sparse as sp

# Import functions directly from megu.py
from .megu import propagate, sparse_mx_to_torch_sparse_tensor, normalize_adj, MeguTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SamplerBase:
    """Base class for sampling strategies in ContrastiveAscentTrainer"""
    
    def __init__(self):
        """Initialize the base sampler class"""
        pass
    
    def sample_points(self, model, data, attacked_idx, args):
        """Sample points based on specific strategy and return sample mask and negative sample dict"""
        if args.request == "edge":
            sample_mask, negative_sample_dict = self.sample_edge_points(model, data, attacked_idx, args)
        else:
            sample_mask, negative_sample_dict = self.sample_node_points(model, data, attacked_idx, args)
        return sample_mask, negative_sample_dict
        
    def sample_node_points(self, model, data, attacked_idx, args):
        """Sample points for node attack scenario"""
        raise NotImplementedError("Implement in subclass")
        
    def sample_edge_points(self, model, data, attacked_idx, args):
        """Sample points for edge attack scenario"""
        raise NotImplementedError("Implement in subclass")
    
    def create_negative_sample_dict(self, poisoned_edges):
        """Create dictionary of negative samples for edge attacks"""
        negative_sample_dict = {}
        
        for i in range(len(poisoned_edges[0])):
            toNode = poisoned_edges[0][i].item()
            fromNode = poisoned_edges[1][i].item()

            if toNode not in negative_sample_dict:
                negative_sample_dict[toNode] = set()
            negative_sample_dict[toNode].add(fromNode)

            if fromNode not in negative_sample_dict:
                negative_sample_dict[fromNode] = set()
            negative_sample_dict[fromNode].add(toNode)
            
        return negative_sample_dict


class LogitDiffTopKSampler(SamplerBase):
    """Identifies influenced nodes by measuring difference between original and perturbed logits using various metrics"""
    
    def __init__(self, metric='l1'):
        """Initialize the logit difference sampler with specified distance metric
        
        Args:
            metric (str): Distance metric to use. Options are:
                'l1': L1 distance (Manhattan)
                'l2': L2 distance (Euclidean)
                'kl': KL-divergence
                'js': Jensen-Shannon divergence
                'cosine': Cosine distance
        """
        super().__init__()
        self.metric = metric.lower()
        print(f"Using {self.metric.upper()} distance metric for logit comparison")
    
    def reverse_features(self, features, idx_to_flip):
        """Helper to flip features of attacked nodes"""
        reverse_features = features.clone() 
        for idx in idx_to_flip:
            reverse_features[idx] = 1 - reverse_features[idx]
        return reverse_features
    
    def calculate_distance(self, logits_a, logits_b):
        """Calculate distance between two sets of logits based on the chosen metric"""
        if self.metric == 'l1':
            # L1/Manhattan distance
            return torch.abs(logits_a - logits_b).mean(dim=1)
        
        elif self.metric == 'l2':
            # L2/Euclidean distance
            return torch.sqrt(torch.sum((logits_a - logits_b) ** 2, dim=1))
        
        elif self.metric == 'kl':
            # KL-divergence (averaged across classes)
            kl_div = F.kl_div(
                F.log_softmax(logits_a, dim=1),
                F.softmax(logits_b, dim=1),
                reduction='none'
            )
            return kl_div.sum(dim=1)
        
        elif self.metric == 'js':
            # Jensen-Shannon divergence
            p = F.softmax(logits_a, dim=1)
            q = F.softmax(logits_b, dim=1)
            m = 0.5 * (p + q)
            
            # Calculate KL(p||m) and KL(q||m)
            kl_p_m = F.kl_div(torch.log(m + 1e-10), p, reduction='none').sum(dim=1)
            kl_q_m = F.kl_div(torch.log(m + 1e-10), q, reduction='none').sum(dim=1)
            
            # JS = 0.5 * (KL(p||m) + KL(q||m))
            return 0.5 * (kl_p_m + kl_q_m)
        
        elif self.metric == 'cosine':
            # Cosine distance (1 - cosine similarity)
            norm_a = F.normalize(logits_a, p=2, dim=1)
            norm_b = F.normalize(logits_b, p=2, dim=1)
            cos_sim = torch.sum(norm_a * norm_b, dim=1)
            return 1 - cos_sim
        
        else:
            # Default to L1 if invalid metric
            print(f"Warning: Unknown metric '{self.metric}'. Using L1 distance instead.")
            return torch.abs(logits_a - logits_b).mean(dim=1)
    
    def sample_node_points(self, model, data, attacked_idx, args):
        """Sample nodes based on distance between original and perturbed logits"""
        # Get k-hop subgraph
        subset, _, _, _ = k_hop_subgraph(
            attacked_idx.clone().detach(), args.k_hop, data.edge_index
        )
        
        # Remove attacked nodes from the subset
        subset = subset[~np.isin(subset.cpu(), attacked_idx.cpu())]
        
        # Get original logits
        og_logits = model(data.x, data.edge_index)
        
        # Get logits with reversed features
        reverse_feature = self.reverse_features(data.x.clone(), attacked_idx)
        final_logits = model(reverse_feature, data.edge_index)
        
        # Calculate distance using the specified metric
        diff = self.calculate_distance(og_logits, final_logits)
        diff = diff[subset]
        
        # Select top nodes by difference
        frac = args.contrastive_frac
        _, indices = torch.topk(diff, int(frac * len(subset)), largest=True)
        influence_nodes = subset[indices]
        
        print(f"Nodes influenced: {len(influence_nodes)} (using {self.metric.upper()} metric)")
        
        # Create sample mask
        sample_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        sample_mask[influence_nodes] = True
        
        return sample_mask, None
    
    def sample_edge_points(self, model, data, attacked_idx, args):
        """Sample nodes based on difference between original and perturbed logits (edge attack)"""
        # Get original logits
        og_logits = model(data.x, data.edge_index)
        
        # Get logits with reversed features of poisoned nodes
        reverse_feature = self.reverse_features(data.x.clone(), data.poisoned_nodes)
        final_logits = model(reverse_feature, data.edge_index)
        
        # Calculate distance using the specified metric
        diff = self.calculate_distance(og_logits, final_logits)
        diff = diff[data.poisoned_nodes]
        
        # Select top nodes by difference
        frac = args.contrastive_frac
        _, indices = torch.topk(diff, int(frac * len(data.poisoned_nodes)), largest=True)
        influence_nodes = data.poisoned_nodes[indices]
        
        print(f"Nodes influenced: {len(influence_nodes)} (using {self.metric.upper()} metric)")
        
        # Create sample mask
        sample_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        sample_mask[influence_nodes] = True
        
        # Create negative sample dict
        poisoned_edges = data.edge_index[:, data.df_mask]
        negative_sample_dict = self.create_negative_sample_dict(poisoned_edges)
        
        return sample_mask, negative_sample_dict


class MeguSampler(SamplerBase):
    """Implementation that directly uses MEGU's original neighbor selection logic"""
    
    def __init__(self):
        """Initialize the MEGU sampler with default values"""
        super().__init__()
    
    def reverse_features(self, features, idx_to_flip):
        """Direct copy of MeguTrainer's reverse_features"""
        reverse_features = features.clone()
        valid_indices = [idx for idx in idx_to_flip if idx < features.shape[0]]
        reverse_features[valid_indices] = 1 - reverse_features[valid_indices]
        return reverse_features
    
    def sample_node_points(self, model, data, attacked_idx, args):
        """Uses MeguTrainer's neighbor_select method directly"""
        try:
            # Create trainer with necessary components for neighbor_select
            # MeguTrainer.__init__ automatically normalizes the adjacency matrix
            temp_trainer = MeguTrainer(model, data, None, args)
            
            # Set temp_node to match the attacked_idx for neighbor_select to work correctly
            if isinstance(attacked_idx, torch.Tensor):
                temp_trainer.temp_node = attacked_idx.cpu().numpy()
            else:
                temp_trainer.temp_node = attacked_idx
                
            # Set num_layers based on args.k_hop instead of hardcoding
            temp_trainer.num_layers = args.k_hop
            
            # Override the default threshold parameters if provided in args
            # These will be used in the neighbor_select method
            if hasattr(args, 'megu_init_alpha'):
                temp_trainer.init_alpha = args.megu_init_alpha
            if hasattr(args, 'megu_gamma'):
                temp_trainer.gamma = args.megu_gamma
            
            # Now we can directly call the original neighbor_select method
            neighbor_nodes_mask = temp_trainer.neighbor_select(data.x)
            
            print(f"Nodes influenced: {neighbor_nodes_mask.sum().item()} (using MEGU's adaptive threshold with {args.k_hop}-hop, init_alpha={temp_trainer.init_alpha}, gamma={temp_trainer.gamma})")
            
            return neighbor_nodes_mask, None
            
        except Exception as e:
            print(f"Error in MeguSampler.sample_node_points: {e}")
            # Fallback to empty mask in case of error
            return torch.zeros(data.num_nodes, dtype=torch.bool), None
    
    def sample_edge_points(self, model, data, attacked_idx, args):
        """Uses MeguTrainer's neighbor_select method directly for edge attacks"""
        try:
            # Create a temporary MeguTrainer instance
            # MeguTrainer.__init__ automatically normalizes the adjacency matrix
            temp_trainer = MeguTrainer(model, data, None, args)
            
            # We explicitly set temp_node for edge attacks
            if hasattr(data, 'poisoned_nodes'):
                if isinstance(data.poisoned_nodes, torch.Tensor):
                    temp_trainer.temp_node = data.poisoned_nodes.cpu().numpy()
                else:
                    temp_trainer.temp_node = data.poisoned_nodes
                    
            # Set num_layers based on args.k_hop instead of hardcoding
            temp_trainer.num_layers = args.k_hop
            
            # Override the default threshold parameters if provided in args
            if hasattr(args, 'megu_init_alpha'):
                temp_trainer.init_alpha = args.megu_init_alpha
            if hasattr(args, 'megu_gamma'):
                temp_trainer.gamma = args.megu_gamma
            
            # Now we can directly call the original neighbor_select method
            neighbor_nodes_mask = temp_trainer.neighbor_select(data.x)
            
            print(f"Nodes influenced: {neighbor_nodes_mask.sum().item()} (using MEGU's adaptive threshold with {args.k_hop}-hop, init_alpha={temp_trainer.init_alpha}, gamma={temp_trainer.gamma})")
            
            # Create negative sample dict
            poisoned_edges = data.edge_index[:, data.df_mask]
            negative_sample_dict = self.create_negative_sample_dict(poisoned_edges)
            
            return neighbor_nodes_mask, negative_sample_dict
            
        except Exception as e:
            print(f"Error in MeguSampler.sample_edge_points: {e}")
            # Fallback to empty mask in case of error
            return torch.zeros(data.num_nodes, dtype=torch.bool), {}

# Legacy class name for backward compatibility
class L1LogitDiffTopKSampler(LogitDiffTopKSampler):
    """Legacy class for backward compatibility"""
    def __init__(self):
        super().__init__(metric='l1')


def get_sampler(strategy_name="megu_sampling", metric='l1'):
    """Factory function to get the appropriate sampler
    
    Args:
        strategy_name (str): Name of the sampling strategy
        metric (str): Distance metric for LogitDiffTopKSampler
            Options: 'l1', 'l2', 'kl', 'js', 'cosine'
    """
    if strategy_name == "logit_diff_topk":
        return LogitDiffTopKSampler(metric=metric)
    elif strategy_name == "l1_logit_diff_topk":
        return L1LogitDiffTopKSampler()
    elif strategy_name == "megu_sampling":
        return MeguSampler()
    else:
        print(f"Warning: Unknown sampling strategy '{strategy_name}'. Using default 'megu_sampling'.")
        return MeguSampler()
