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

unreachable_distance = -1

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
        self.real_anomaly_ratio = 0.0

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
        df_graph_labels = pd.read_csv(filename_graph_labels)
        labels = df_graph_labels['label'].to_numpy().astype(np.int32)
        sampled_labels = []
        sampled_anomalies_indices = np.array([])
        if self.required_anomaly_ratio == 0.0:
            self.real_anomaly_ratio = 0.0
            self.anomaly_num = 0
        else:  # self.required_anomaly_ratio > 0:
            sampled_anomalies_indices, self.real_anomaly_ratio = sampling_anomalies_by_ratio(labels, self.required_anomaly_ratio)
            self.anomaly_num = len(sampled_anomalies_indices)
        del df_graph_labels

        # load node attributes
        df_node_attrs = pd.read_csv(filename_node_attrs, header=None)
        self.dataset_node_size, self.feat_dim = df_node_attrs.shape
        if self.feature_type == 'semantics':
            feat = torch.from_numpy(df_node_attrs.to_numpy().astype(np.float32))
        else:
            feat = nn.Embedding(self.dataset_node_size, self.feat_dim).weight
        del df_node_attrs

        df_edges_data = pd.read_csv(filename_edge_info)
        # For the edges, first group the table by graph IDs.
        edges_group = df_edges_data.groupby('graph_idx')

        for i, graph_id in enumerate(tqdm(edges_group.groups, desc="Calculating nodes")):
            if (self.required_anomaly_ratio == 0.0 and labels[i] == 1) or \
                    (labels[i] == 1 and (i not in sampled_anomalies_indices)):
                continue
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            current_nodes_num = len(np.unique(np.concatenate((src, dst))))
            if current_nodes_num > self.max_graph_node_size:
                self.max_graph_node_size = current_nodes_num

            del edges_of_id, src, dst

        for i, graph_id in enumerate(tqdm(edges_group.groups, desc="Loading data")):
            if (self.required_anomaly_ratio == 0.0 and labels[i] == 1) or \
                    (labels[i] == 1 and (i not in sampled_anomalies_indices)):
                continue
            sampled_labels.append(labels[i])
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            # convert all the index to start from 0
            src = torch.from_numpy(edges_of_id['src'].to_numpy() - 1)
            dst = torch.from_numpy(edges_of_id['dst'].to_numpy() - 1)
            weight = torch.from_numpy(edges_of_id['weight'].to_numpy())

            graph = dgl.graph((src, dst))
            graph.ndata['feat'] = feat[:torch.max(torch.cat((src, dst)))+1, :]
            graph.edata['weight'] = weight

            begin_time = time.time()
            # remove those nodes without connection to other nodes
            remove_nodes = [node for node in graph.nodes() if graph.out_degrees(node) == 0 and graph.in_degrees(node) == 0]
            graph.remove_nodes(remove_nodes)
            # print(f"remove {len(remove_nodes)} nodes")

            remove_nodes_elapsed = time.time() - begin_time

            # add self_loop to ensure the self_node feature convoluted and set their edge weights as 1
            # graph = dgl.add_self_loop(graph)
            # graph.edata['weight'][graph.edata['weight'] == 0] = 1

            # begin_time = time.time()
            # dist_matrix = shortest_path(csgraph=graph.adj(scipy_fmt='csr'), directed=True)
            # # build the distance attribute for nodes
            # graph.ndata['distance'] = torch.from_numpy(dist_matrix)
            # distance_calc_elapsed = time.time() - begin_time
            #
            # # replace unreachable distance 'inf' with a certain value( current is -1) across graphs
            # graph.ndata['distance'][graph.ndata['distance'] == float('Inf')] = unreachable_distance
            # graph.ndata['distance'] = graph.ndata['distance'].type(torch.long)

            if self.verbose:
                print(f"Finished graph {graph_id} data loading and preprocessing.")
                print(f"  NumNodes: {graph.num_nodes()}")
                print(f"  NumEdges: {graph.num_edges()}")
                print(f"  Feat_dim: {graph.ndata['feat'].shape[1]}")

                # print(f"Memories allocation: {100.0 * (mem1 - mem0) / mem0:.2f}")
                # print(f"shortest_path calculation consumes {distance_calc_elapsed:.3f)}(s)")
                # print(f"remove nodes consumes {remove_nodes_elapsed:.3f}(s)")

            self.graphs.append(graph)

            # free allocated memory that will not be used anymore
            del edges_of_id, src, dst, weight, graph, remove_nodes  # , dist_matrix

            if (i + 1) % 1000 == 0:
                gc.collect()

            # mem1 = proc.memory_info().rss
        self.labels = torch.LongTensor(np.array(sampled_labels))
        assert len(self.graphs) == len(self.labels)
        if self.verbose:
            print(f"  NumGraphs in this dataset: {len(self.graphs)}")

        del df_edges_data

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
                              'real_anomaly_ratio': self.real_anomaly_ratio,
                              'anomaly_num': self.anomaly_num})

    def load(self):
        graph_path = os.path.join(self.save_dir, f"{self.data_name}_{self.supervise_mode}_{self.required_anomaly_ratio}{self.prefix}_graph.bin")
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        self.feat_dim = self.graphs[0].ndata['feat'].shape[1]
        info_path = os.path.join(self.save_dir, f"{self.data_name}_{self.supervise_mode}_{self.required_anomaly_ratio}{self.prefix}_info.bin")
        info = load_info(info_path)
        self.dataset_node_size = info['dataset_node_size']
        self.max_graph_node_size = info['max_graph_node_size']
        self.real_anomaly_ratio = info['real_anomaly_ratio']
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
    if (float(anomaly_num/len(data))) > anomaly_ratio:
        for count in range(1, anomaly_num):
            sampled_ratio = float(count / (count + normal_num))
            if sampled_ratio >= anomaly_ratio:
                sampled_anomaly_indices = np.random.choice(anomaly_indices[0], size=count, replace=False)
                return sampled_anomaly_indices, sampled_ratio
    else:
        return anomaly_indices[0], float(anomaly_num/len(data))


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
