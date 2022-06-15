# -*- coding: utf-8 -*-
"""
Created on Tue April 07 17:16:36 2022

@author: Yongzheng Xie
"""
import dgl
import numpy as np
import torch
import torch.utils.data
from torch_geometric.data import Data
from LogGraphDataset import unreachable_distance
from tqdm import tqdm
import util


graph_indicator = 0
label_indicator = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphFeatureExtractor(torch.utils.data.Dataset):
    ''' graphs and nodes in graph
    '''

    def __init__(self, input_graphs, max_dataset_nodes=0, max_graph_nodes=0, node_attr_dim=0, transform=None):

        self.data_list = []
        self.max_dataset_nodes = max_dataset_nodes
        self.max_num_nodes = max_graph_nodes
        self.feat_dim = node_attr_dim
        self.transform = transform

        print(f"Start feature extracting....")
        for (graph, label) in tqdm(input_graphs):
            f = torch.zeros((self.max_num_nodes, self.feat_dim), dtype=torch.float32)
            in_degree = torch.zeros(self.max_num_nodes, dtype=torch.int32)
            out_degree = torch.zeros(self.max_num_nodes, dtype=torch.int32)

            for i, u in enumerate(graph.nodes()):
                f[i, :] = graph.ndata['feat'][u]
                in_degree[i] = graph.in_degrees(u)
                out_degree[i] = graph.out_degrees(u)

            data = Data(x=f,
                        edge_index=graph.edges(),
                        # all the edges share the same attributes
                        edge_attr=torch.ones_like(graph.edata['weight']),
                        y=label
                        )
            data['edge_weight'] = graph.edata['weight']
            data['in_degree'] = in_degree
            data['out_degree'] = out_degree
            data['num_nodes'] = graph.num_nodes()

            if self.transform is not None:
                self.data_list.append(self.transform(data))
            else:
                self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
