# import networkx as nx
import numpy as np
import torch

class GraphData(torch.utils.data.Dataset):
    def __init__(self, adj_all, features_all, u_all, labels_all, max_num_nodes, padded=False, index=None):
        # the input adj_all must be unpadded
        self.max_num_nodes = max_num_nodes
        self.adj_all = adj_all
        self.len_all = []  # num of nodes (original, not padded)
        self.feature_all = features_all
        self.u_all = u_all
        self.labels_all = labels_all
        self.padded = padded
        if index is None:
            self.index = [i for i in range(len(self.adj_all))]
        else:
            self.index = index

        for adj in adj_all:
            self.len_all.append(adj.shape[0])
        if padded:
            self.adj_all = padding_graphs(adj_all, max_num_nodes)
            self.feature_all = padding_features(features_all, max_num_nodes)

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = self.len_all[idx]

        return {'adj': adj.copy(),
                'features': self.feature_all[idx].copy(),
                'labels': self.labels_all[idx].copy(),
                'u': self.u_all[idx].copy(),
                'num_node_real': self.len_all[idx],
                'index': self.index[idx]
                }


def padding_graphs(adj_all, max_num_nodes):
    adj_all_padded = []
    for adj in adj_all:
        num_nodes = adj.shape[0]
        adj_padded = np.eye((max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        adj_all_padded.append(adj_padded)

    return adj_all_padded

def padding_features(features_all, max_num_nodes):
    features_all_padded = []
    feat_dim = features_all[0].shape[1]
    for features in features_all:
        num_nodes = features.shape[0]
        features_padded = np.zeros((max_num_nodes, feat_dim))
        features_padded[:num_nodes] = features
        features_all_padded.append(features_padded)
    return features_all_padded