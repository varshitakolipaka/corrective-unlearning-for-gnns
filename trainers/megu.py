import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

torch.cuda.empty_cache()
from torch_geometric.nn import CorrectAndSmooth
import numpy as np
from trainers.node_classifier import NodeClassifier
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
import scipy.sparse as sp
from trainers.megu_utils import calc_f1
import logging
from .base import Trainer

class GATE(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lr = torch.nn.Linear(dim, dim)

    def forward(self, x):
        t = x.clone()
        return self.lr(t)


def criterionKD(p, q, T=1.5):
    loss_kl = nn.KLDivLoss(reduction="batchmean")
    soft_p = F.log_softmax(p / T, dim=1)
    soft_q = F.softmax(q / T, dim=1).detach()
    return loss_kl(soft_p, soft_q)


def propagate(features, k, adj_norm):
    feature_list = []
    feature_list.append(features)
    for i in range(k):
        feature_list.append(torch.spmm(adj_norm, feature_list[-1]))
    return feature_list[-1]


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_adj(adj, r=0.5):
    adj = adj + sp.eye(adj.shape[0])
    degrees = np.array(adj.sum(1))
    r_inv_sqrt_left = np.power(degrees, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)

    r_inv_sqrt_right = np.power(degrees, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)

    adj_normalized = adj.dot(r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right)
    return adj_normalized


class ExpMEGU(Trainer):
    def __init__(self, model, data, optimizer, args):
        super().__init__(model, data, optimizer)

        self.logger = logging.getLogger('exp')
        self.args = args
        self.logger = logging.getLogger('ExpMEGU')
        self.data = data # instead of using load data
        self.model = model # poisoned model 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_feats = self.data.num_features
        self.train_indices = torch.where(self.data.train_mask)[0]
        self.test_indices = torch.where(self.data.test_mask)[0]

        self.unlearning_request()

        self.target_model_name = self.args.gnn
        self.determine_target_model()

        self.num_layers = 2
        self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj(to_scipy_sparse_matrix(self.data.edge_index)))
        self.neighbor_khop = self.neighbor_select(self.data.x)

        self.target_model.model = self.model
        dt_acc, msc_rate, dt_f1 = self.evaluate()
        print("the poisoned model: ")
        print(dt_acc, msc_rate, dt_f1)

        self.train()
        self.model = self.target_model.model
        
        dt_acc, msc_rate, dt_f1 = self.evaluate()
        print("the unlearnt model: ")
        print(dt_acc, msc_rate, dt_f1)

    def unlearning_request(self):

        self.data.x_unlearn = self.data.x.clone()
        self.data.edge_index_unlearn = self.data.edge_index.clone()
        edge_index = self.data.edge_index.numpy()
        unique_indices = np.where(edge_index[0] < edge_index[1])[0]

        if self.args.request == 'node':
            unique_nodes = torch.where(self.data.df_mask)[0]
            self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes)

        if self.args.request == 'edge':
            remove_indices = torch.where(self.data.df_mask)[0]
            remove_edges = edge_index[:, remove_indices]
            unique_nodes = np.unique(remove_edges)

            self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes, remove_indices)

        self.temp_node = unique_nodes

    def update_edge_index_unlearn(self, delete_nodes, delete_edge_index=None):
        edge_index = self.data.edge_index.numpy()

        unique_indices = np.where(edge_index[0] < edge_index[1])[0]
        unique_indices_not = np.where(edge_index[0] > edge_index[1])[0]

        if self.args.request == 'edge':
            remain_indices = np.setdiff1d(unique_indices, delete_edge_index)
        else:
            unique_edge_index = edge_index[:, unique_indices]
            delete_edge_indices = np.logical_or(np.isin(unique_edge_index[0], delete_nodes),
                                                np.isin(unique_edge_index[1], delete_nodes))
            remain_indices = np.logical_not(delete_edge_indices)
            remain_indices = np.where(remain_indices == True)[0]

        remain_encode = edge_index[0, remain_indices] * edge_index.shape[1] * 2 + edge_index[1, remain_indices]
        unique_encode_not = edge_index[1, unique_indices_not] * edge_index.shape[1] * 2 + edge_index[0, unique_indices_not]
        sort_indices = np.argsort(unique_encode_not)

        print(f"sort_indices: {sort_indices}")
        print(f"unique_encode_not: {unique_encode_not}")
        print(f"remain_encode: {remain_encode}")

        indices_to_check = np.searchsorted(unique_encode_not, remain_encode, sorter=sort_indices)
        valid_indices = indices_to_check[indices_to_check < unique_encode_not.size]
        valid_remain_encode = remain_encode[indices_to_check < unique_encode_not.size]

        remain_indices_not = unique_indices_not[
            sort_indices[np.searchsorted(unique_encode_not, valid_remain_encode, sorter=sort_indices)]]
        remain_indices = np.union1d(remain_indices, remain_indices_not)

        return torch.from_numpy(edge_index[:, remain_indices])

    def determine_target_model(self):
        # self.logger.info('target model: %s' % (self.args['target_model'],))
        num_classes = self.data.num_classes

        self.target_model = NodeClassifier(self.num_feats, num_classes, self.args, self.data)



    def neighbor_select(self, features):
        temp_features = features.clone()
        pfeatures = propagate(temp_features, self.num_layers, self.adj)
        reverse_feature = self.reverse_features(temp_features)
        re_pfeatures = propagate(reverse_feature, self.num_layers, self.adj)

        cos = nn.CosineSimilarity()
        sim = cos(pfeatures, re_pfeatures)
        
        alpha = 0.1
        gamma = 0.1
        max_val = 0.
        while True:
            influence_nodes_with_unlearning_nodes = torch.nonzero(sim <= alpha).flatten().cpu()
            if len(influence_nodes_with_unlearning_nodes.view(-1)) > 0:
                temp_max = torch.max(sim[influence_nodes_with_unlearning_nodes])
            else:
                alpha = alpha + gamma
                continue

            if temp_max == max_val:
                break

            max_val = temp_max
            alpha = alpha + gamma

        # influence_nodes_with_unlearning_nodes = torch.nonzero(sim < 0.5).squeeze().cpu()
        neighborkhop, _, _, two_hop_mask = k_hop_subgraph(
            torch.tensor(self.temp_node),
            self.num_layers,
            self.data.edge_index,
            num_nodes=self.data.num_nodes)

        neighborkhop = neighborkhop[~np.isin(neighborkhop.cpu(), self.temp_node)]
        neighbor_nodes = []
        for idx in influence_nodes_with_unlearning_nodes:
            if idx in neighborkhop and idx not in self.temp_node:
                neighbor_nodes.append(idx.item())
        
        neighbor_nodes_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), neighbor_nodes))

        return neighbor_nodes_mask

    def reverse_features(self, features):
        reverse_features = features.clone()
        for idx in self.temp_node:
            reverse_features[idx] = 1 - reverse_features[idx]

        return reverse_features

    def correct_and_smooth(self, y_soft, preds):
        pos = CorrectAndSmooth(num_correction_layers=80, correction_alpha=self.args.alpha1,
                               num_smoothing_layers=80, smoothing_alpha=self.args.alpha2,
                               autoscale=False, scale=1.)

        y_soft = pos.correct(y_soft, preds[self.data.train_mask], self.data.train_mask,
                                  self.data.edge_index_unlearn)
        y_soft = pos.smooth(y_soft, preds[self.data.train_mask], self.data.train_mask,
                                 self.data.edge_index_unlearn)

        return y_soft

    def train(self):
        operator = GATE(self.data.num_classes).to(self.device)

        optimizer = torch.optim.SGD([
            {'params': self.target_model.model.parameters()},
            {'params': operator.parameters()}
        ], lr=self.args.unlearn_lr)

        with torch.no_grad():
            self.target_model.model.eval()
            preds = self.target_model.model(self.data.x, self.data.edge_index)
            if self.args.dataset == 'ppi':
                preds = torch.sigmoid(preds).ge(0.5)
                preds = preds.type_as(self.data.y)
            else:
                preds = torch.argmax(preds, axis=1).type_as(self.data.y)


        start_time = time.time()
        for epoch in range(self.args.unlearning_epochs):
            self.target_model.model.train()
            operator.train()
            optimizer.zero_grad()
            out_ori = self.target_model.model(self.data.x_unlearn, self.data.edge_index_unlearn)
            out = operator(out_ori)

            if self.args.dataset == 'ppi':
                loss_u = criterionKD(out_ori[self.temp_node], out[self.temp_node]) - F.binary_cross_entropy_with_logits(out[self.temp_node], preds[self.temp_node])
                loss_r = criterionKD(out[self.neighbor_khop], out_ori[self.neighbor_khop]) + F.binary_cross_entropy_with_logits(out_ori[self.neighbor_khop], preds[self.neighbor_khop])
            else:
                loss_u = criterionKD(out_ori[self.temp_node], out[self.temp_node]) - F.cross_entropy(out[self.temp_node], preds[self.temp_node])
                loss_r = criterionKD(out[self.neighbor_khop], out_ori[self.neighbor_khop]) + F.cross_entropy(out_ori[self.neighbor_khop], preds[self.neighbor_khop])

            loss = self.args.kappa * loss_u + loss_r

            loss.backward()
            optimizer.step()

        unlearn_time = time.time() - start_time
        self.target_model.model.eval()
        test_out = self.target_model.model(self.data.x_unlearn, self.data.edge_index_unlearn)
        if self.args.dataset == 'ppi':
            out = torch.sigmoid(test_out)
        else:
            out = self.correct_and_smooth(F.softmax(test_out, dim=-1), preds)

        y_hat = out.cpu().detach().numpy()
        y = self.data.y.cpu()
        if self.args.dataset == 'ppi':
            test_f1 = calc_f1(y, y_hat, self.data.test_mask, multilabel=True)
        else:
            test_f1 = calc_f1(y, y_hat, self.data.test_mask)


        return unlearn_time, test_f1
