# -*- coding: utf-8 -*-
"""
Created on Tue April 07 17:16:36 2022

@author: Yongzheng Xie
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import gc
import psutil
import re
import torch
import torch.nn as nn
import dgl
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from scipy.sparse.csgraph import shortest_path
import torch.nn.functional as F
from dgl.data.utils import load_info, save_info

import warnings

warnings.filterwarnings("ignore", message="DGLGraph.__len__")

unreachable_distance = 20.0

class LogGraphDataset(DGLDataset):

    def __init__(self, data_name, supervise_mode, window_size, feature_type='non_semantics', use_tfidf=False, transform=None, url=None,
                 raw_dir=None, save_dir=None, dataset_type=None, anomaly_ratio=1.0, force_reload=False, verbose=False):
        assert data_name.lower() in ['hdfs', 'bgl', 'spirit', 'tbd']
        self.data_name = data_name
        self.supervise_mode = supervise_mode
        self.dataset_node_size = 0
        self.max_graph_node_size = 0
        self.root = raw_dir
        self.graphs = []
        self.labels = []
        self.num_labels = 2
        self.feat_dim = None
        self.use_tfidf = use_tfidf
        self.feature_type = feature_type
        self.prefix = "" if dataset_type is None else f"_{dataset_type}"
        self.prefix += f'_{window_size}_{feature_type}'
        self.transform = transform
        self.anomaly_num = 0
        assert anomaly_ratio >= 0.0
        self.required_anomaly_ratio = anomaly_ratio
        self.real_anomaly_ratio = 0

        super(LogGraphDataset, self).__init__(name=data_name,
                                              url=url,
                                              raw_dir=raw_dir,
                                              save_dir=save_dir,
                                              force_reload=force_reload,
                                              verbose=verbose)

    def process(self):
        # monitor the usage of memory
        # proc = psutil.Process(os.getpid())
        # mem0 = proc.memory_info().rss
        tfidf_prefix = '_tfidf' if self.use_tfidf else ''
        filename_graph_labels = f"{self.root}/{self.data_name}{self.prefix}_labels.csv"
        filename_node_attrs = f"{self.root}/{self.data_name}{tfidf_prefix}_node_attributes.csv"
        filename_edge_info = f"{self.root}/{self.data_name}{self.prefix}_edges.csv"

        if not (os.path.exists(filename_edge_info) or
                os.path.exists(filename_node_attrs) or
                os.path.exists(filename_graph_labels)):
            print(f"The specified file does not exist!")
            return None

        # load graph label data
        df = pd.read_csv(filename_graph_labels)
        labels = df['label'].to_numpy().astype(np.int32)
        sampled_labels = []
        if self.required_anomaly_ratio == 0.0:
            self.real_anomaly_ratio = 0.0
            self.anomaly_num = 0
        else:  # self.required_anomaly_ratio > 0:
            sampled_anomalies_indices, self.real_anomaly_ratio = sampling_anomalies_by_ratio(labels, self.required_anomaly_ratio)
            self.anomaly_num = len(sampled_anomalies_indices)

        del df

        # load node attributes
        node_attrs = {}
        node_idx = 0
        try:
            with open(filename_node_attrs) as f:
                for line in f:
                    line = line.strip("\n")
                    attrs = [float(attr) for attr in re.split("[,]+", line) if not attr == '']
                    node_attrs[node_idx] = np.array(attrs)
                    node_idx += 1
                self.dataset_node_size = node_idx
                if self.verbose:
                    print(f"The number of nodes shared by each graph= {len(node_attrs)}")
        except IOError:
            print('No node attributes')
        if self.feature_type == 'semantics':
            feat = np.array(list(node_attrs.values()))
        else:
            feat = nn.Embedding(node_idx, len(node_attrs[0])).weight
            feat.requires_grad = False

        edges_data = pd.read_csv(filename_edge_info)
        # For the edges, first group the table by graph IDs.
        edges_group = edges_data.groupby('graph_idx')

        for i, graph_id in enumerate(tqdm(edges_group.groups)):
            if (self.required_anomaly_ratio == 0 and labels[i] == 1) or (labels[i] == 1 and (i not in sampled_anomalies_indices)):
                continue
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            current_nodes_num = len(np.unique(np.concatenate((src, dst))))
            if current_nodes_num > self.max_graph_node_size:
                self.max_graph_node_size = current_nodes_num

        for i, graph_id in enumerate(tqdm(edges_group.groups)):
            if (self.required_anomaly_ratio == 0 and labels[i] == 1) or (labels[i] == 1 and (i not in sampled_anomalies_indices)):
                continue
            sampled_labels.append(labels[i])
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            # convert all the index to start from 0
            src = torch.from_numpy(edges_of_id['src'].to_numpy() - 1)
            dst = torch.from_numpy(edges_of_id['dst'].to_numpy() - 1)
            weight = edges_of_id['weight'].to_numpy().astype(np.float32)

            # use the maximum node number to create the graph
            graph = dgl.graph((src, dst), num_nodes=self.dataset_node_size)
            # graph = dgl.graph((src, dst), idtype=torch.int32)
            # feat = np.array([node_attrs[i.item()] for i in graph.nodes()])
            if self.feature_type == 'semantics':
                graph.ndata['feat'] = torch.tensor(feat, dtype=torch.float32)
            else:
                graph.ndata['feat'] = feat
            graph.edata['weight'] = torch.tensor(weight, dtype=torch.float32)

            begin_time = time.time()
            dist_matrix = shortest_path(csgraph=graph.adj(scipy_fmt='csr'), directed=True)
            distance_calc_elapsed = time.time() - begin_time
            # replace unreachable distance 'inf' with a certain value( current is 20) across graphs
            dist_matrix = np.where(dist_matrix == float('Inf'), unreachable_distance, dist_matrix)

            # unreachable_distance = torch.nan_to_num(graph.ndata['distance'], posinf=20)
            # dist_matrix = np.pad(dist_matrix, ((0, 0), (0, self.max_graph_node_size-graph.num_nodes())), 'constant', constant_values=float('Inf'))
            # dist_matrix[dist_matrix == float('Inf')] = -float('Inf')

            # build the distance attribute for nodes
            graph.ndata['distance'] = torch.from_numpy(dist_matrix.astype(np.float32))

            begin_time = time.time()

            # remove those nodes without connection to other nodes
            # remove_nodes = np.setdiff1d(np.array(graph.nodes().tolist()), np.unique(np.concatenate((src, dst))))

            remove_nodes = [node for node in graph.nodes() if graph.out_degrees(node) == 0 and graph.in_degrees(node) == 0]
            graph.remove_nodes(remove_nodes)
            # print(f"remove {len(remove_nodes)} nodes")
            if graph.num_nodes() > self.max_graph_node_size:
                self.max_graph_node_size = graph.num_nodes()
            remove_elapsed = time.time() - begin_time

            # pad the dimension of node['distance'] in each graph to the same size
            # new_node_num = self.max_graph_node_size-graph.num_nodes()
            # graph = dgl.add_nodes(graph, new_node_num, {'distance': torch.tensor([unreachable_distance], dtype=torch.int32).repeat(new_node_num, self.dataset_node_size)})

            # add self_loop to ensure the self_node feature convoluted
            graph = dgl.add_self_loop(graph)
            # The edge weight for self_loop is set as 1
            graph.edata['weight'][graph.edata['weight'] == 0] = 1.0
            graph.edata['weight'] = F.normalize(graph.edata['weight'],dim=0, p=1)
            graph.ndata['distance'] = F.normalize(graph.ndata['distance'],dim=1, p=1)
            if self.verbose:
                print(f"Finished graph {graph_id} data loading and preprocessing.")
                print(f"  NumNodes: {graph.num_nodes()}")
                print(f"  NumEdges: {graph.num_edges()}")
                print(f"  Feat_dim: {graph.ndata['feat'].shape[1]}")

                # print(f"Memories allocation: {100.0 * (mem1 - mem0) / mem0:.2f}")
                print(f"shortest_path calculation consumes {distance_calc_elapsed:.3f)}(s)")
                print(f"remove nodes consumes {remove_elapsed:.3f}(s)")

            self.graphs.append(graph)

            # free allocated memory that will not be used anymore
            del edges_of_id, src, dst, weight, dist_matrix, remove_nodes, graph
            if (i + 1) % 1000 == 0:
                gc.collect()

            # mem1 = proc.memory_info().rss
        self.labels = torch.tensor(np.array(sampled_labels), dtype=torch.int32)
        self.feat_dim = self.graphs[0].ndata['feat'].shape[1]
        assert len(self.graphs) == len(self.labels)
        if self.verbose:
            print(f"  NumGraphs in this dataset: {len(self.graphs)}")

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        # Save graphs and the corresponding labels
        graph_path = os.path.join(self.save_dir, f"{self.data_name}_{self.supervise_mode}_{self.required_anomaly_ratio}{self.prefix}_graph.bin")
        save_graphs(graph_path, self.graphs, {'labels': self.labels})
        # Save graph info
        info_path = os.path.join(self.save_dir, f"{self.data_name}_{self.supervise_mode}_{self.required_anomaly_ratio}{self.prefix}_info.bin")
        save_info(info_path, {'max_graph_node_size': self.max_graph_node_size,
                              'dataset_node_size': self.dataset_node_size,
                              'anomaly_num': self.anomaly_num})

    def load(self):
        graph_path = os.path.join(self.save_dir, f"{self.data_name}_{self.supervise_mode}_{self.required_anomaly_ratio}{self.prefix}_graph.bin")
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        self.feat_dim = self.graphs[0].ndata['feat'].shape[1]
        info_path = os.path.join(self.save_dir, f"{self.data_name}_{self.supervise_mode}_{self.required_anomaly_ratio}{self.prefix}_info.bin")
        info = load_info(info_path)
        self.max_graph_node_size = info['max_graph_node_size']
        self.dataset_node_size = info['dataset_node_size']
        self.anomaly_num = info['anomaly_num']

    def has_cache(self):
        graph_path = os.path.join(self.save_dir, f"{self.data_name}_{self.supervise_mode}_{self.required_anomaly_ratio}{self.prefix}_graph.bin")
        return os.path.exists(graph_path)

def _collate_fn(batch):
    # batch is a list of tuple (graph, label)
    graphs = [e[0] for e in batch]
    g_batch = dgl.batch(graphs)
    g_labels = [e[1] for e in batch]
    g_labels = torch.stack(g_labels, 0)
    return g_batch, g_labels

def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

def sampling_anomalies_by_ratio(input_data, anomaly_ratio):
    data = input_data if isinstance(input_data, np.ndarray) else np.array(input_data)
    anomaly_num = data.sum()
    normal_num = len(data) - anomaly_num
    anomaly_indices = np.where(data == 1)
    if (anomaly_num/len(data)) > anomaly_ratio:
        for count in range(1, anomaly_num):
            sampled_ratio = count / (count + normal_num)
            if sampled_ratio >= anomaly_ratio:
                sampled_anomaly_indices = np.random.choice(anomaly_indices[0], size=count, replace=False)
                return sampled_anomaly_indices, sampled_ratio
    else:
        return anomaly_indices[0], anomaly_num/len(data)

if __name__ == '__main__':

    dataset_list = [{'dataset': 'HDFS', 'data_path': './dataset', 'node_num': 18, 'window_size': 20, 'feature_type': 'semantics'},
                    {'dataset': 'BGL', 'data_path': './dataset', 'node_num': 122, 'window_size': 20, 'feature_type': 'semantics'},
                    ]
    for data_set in dataset_list:
        print(f"starting processing")
        # graph_data = LogGraphDataset(data_name=data_set['dataset'], num_nodes=data_set['node_num'],
        #                              window_size=data_set['window_size'], feature_type=data_set['feature_type'],
        #                              raw_dir=data_set['data_path'], dataset_type='test', verbose=False)
        # data_loader = GraphDataLoader(graph_data, batch_size=32, shuffle=True, collate_fn=_collate_fn)
        #
        # # training
        # for epoch in range(100):
        #     for g, labels in data_loader:
        #         # your training code here
        #         pass
