# -*- coding: utf-8 -*-
"""
Created on Tue April 07 17:16:36 2022

@author: Yongzheng Xie
"""
import math
import dgl
import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch_geometric.utils import remove_isolated_nodes
from torch_geometric.data import Data
from tqdm import tqdm


class GraphFeatureExtractor(torch.utils.data.Dataset):
    ''' graphs and nodes in graph
    '''

    def __init__(self, root, data_name, embedding_type, dataset_type, window_size, feature_type, transform=None, args=None):

        self.root = root
        self.data_name = data_name
        self.embedding_type = embedding_type
        self.dataset_type = dataset_type
        self.window_size = window_size
        self.feature_type = feature_type
        self.prefix = "" if dataset_type is None else f"_{dataset_type}"
        self.prefix += f'_{window_size}_{feature_type}'
        self.embedding_prefix = f'_{embedding_type}'

        self.data_list = []
        self.max_graph_node_size = 0
        self.transform = transform

        filename_graph_labels = f"{root}/{data_name}{self.prefix}_labels.csv"
        filename_node_attrs = f"{root}/{data_name}_node_attributes{self.embedding_prefix}_{dataset_type}_{window_size}.csv"
        filename_edge_info = f"{root}/{data_name}{self.prefix}_edges.csv"

        if not (os.path.exists(filename_edge_info) or os.path.exists(filename_node_attrs) or os.path.exists(filename_graph_labels)):
            print(f"The specified file does not exist!")

        # load graph label data
        df_graph_labels = pd.read_csv(filename_graph_labels)
        labels = df_graph_labels['label'].to_numpy().astype(np.int32)

        self.labels = torch.LongTensor(np.array(labels))
        self.anomaly_num = np.sum(labels)
        self.real_anomaly_ratio = round(self.anomaly_num / len(labels), 3)
        del df_graph_labels

        # load node attributes
        df_node_attrs = pd.read_csv(filename_node_attrs, header=None)
        self.dataset_node_size = df_node_attrs.shape[0] - 1    # excluding the first embedding that represents OOV event
        self.feat_dim = df_node_attrs.shape[1]
        feat = torch.from_numpy(df_node_attrs.to_numpy().astype(np.float32))
        del df_node_attrs

        df_edges_data = pd.read_csv(filename_edge_info)
        # For the edges, first group the table by graph IDs.
        edges_group = df_edges_data.groupby('graph_idx')

        for i, graph_id in enumerate(tqdm(edges_group.groups, desc="Calculating nodes")):
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy() - 1
            dst = edges_of_id['dst'].to_numpy() - 1
            current_nodes_num = len(np.unique(np.concatenate((src, dst))))
            if current_nodes_num > self.max_graph_node_size:
                self.max_graph_node_size = current_nodes_num

            del edges_of_id, src, dst

        for i, graph_id in enumerate(tqdm(edges_group.groups, desc="Loading data")):

            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            # convert all the index to start from 1
            # because 0 indicates padding and 1 indicates OOV, log event index starts from 2 when graph data generating
            src = torch.from_numpy(edges_of_id['src'].to_numpy() - 1)
            dst = torch.from_numpy(edges_of_id['dst'].to_numpy() - 1)
            weight = torch.from_numpy(edges_of_id['weight'].to_numpy())

            if len(src) == 1 and src == dst:
                edge_index = torch.tensor(([0], [0]))
                edge_weight = weight
                node_mask = torch.zeros(feat.shape[0], dtype=torch.bool)
                node_mask[src] = 1
            else:
                # remove those isolated nodes without the connection to other nodes
                (edge_index, edge_weight, node_mask) = remove_isolated_nodes(edge_index=torch.stack((src, dst)),
                                                                             edge_attr=weight,
                                                                             num_nodes=feat.shape[0])

            data = Data(x=feat[node_mask],
                        edge_index=edge_index,
                        # all the edges share the same attributes
                        edge_attr=edge_weight,
                        y=self.labels[i]
                        )

            data['edge_weight'] = edge_weight.to(torch.float32)

            if self.transform is not None:
                self.data_list.append(self.transform(data))
            else:
                self.data_list.append(data)

            # free allocated memory that will not be used anymore
            del edges_of_id, src, dst, weight, edge_index, edge_weight, node_mask, data

        del df_edges_data

    def get_labels(self, idx=None):
        if idx is None:
            return self.labels
        else:
            return self.labels[idx]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def get_samples(self, sample_size=1.0, anomaly_ratio=1.0, shuffle=True, graph_augment=False):
        assert (anomaly_ratio >= 0.0) and (anomaly_ratio <= 1.0)
        assert (math.ceil(sample_size) > 0.0) or (sample_size == -1.0)

        if sample_size == 1.0 and anomaly_ratio == 1.0:
            if graph_augment:
                filtered_labels = [graph.y for graph in self.data_list if (graph.num_nodes > 1)]
                filtered_graphs = [graph for graph in self.data_list if (graph.num_nodes > 1)]
                print(f"{len(filtered_labels)}/{len(self.labels)} graphs that contain the specified unique nodes")
            else:
                filtered_labels = self.labels
                filtered_graphs = self.data_list

            sampled_data = [g for g in filtered_graphs]
            sampled_labels = filtered_labels

            return sampled_data, sampled_labels

        if sample_size == -1.0:
            if (self.anomaly_num * 2) > len(self.labels):
                n_total_samples = 2 * self.anomaly_num
                n_anomaly_samples = self.anomaly_num
            else:
                # n_anomaly_samples = len(self.graphs) - self.anomaly_num
                # n_total_samples = 2 * n_anomaly_samples
                n_total_samples = len(self.data_list)
                if 0.3 < self.real_anomaly_ratio < 0.5:
                    n_anomaly_samples = self.anomaly_num
                else:
                    n_anomaly_samples = int(len(self.data_list) * 0.3)

            if n_total_samples > len(self.labels):
                print(f"Anomalies are oversampling.")
        else:
            if 0.0 < sample_size <= 1.0:
                n_total_samples = int(len(self.labels) * sample_size)
            elif sample_size > len(self.labels):
                n_total_samples = len(self.labels)
                print(f"The sampling size exceeds the size of dataset. The dataset size is used instead of sampling size!")
            else:
                n_total_samples = int(sample_size)

            if anomaly_ratio == 1.0:
                n_anomaly_samples = int(self.real_anomaly_ratio * n_total_samples)
            else:
                n_anomaly_samples = int(anomaly_ratio * n_total_samples)

            if n_anomaly_samples > self.anomaly_num:
                print(f"The anomaly ratio exceeds the dataset anomaly ratio. oversampling is used.")

        sampled_anomalies_indices, filtered_graphs, filtered_labels = self.sampling_by_count(n_anomaly_samples, target_label=1, graph_augment=graph_augment)
        sampled_normal_indices, _, _ = self.sampling_by_count(n_total_samples - n_anomaly_samples, target_label=0, graph_augment=graph_augment)
        sample_indices = np.concatenate((sampled_anomalies_indices, sampled_normal_indices))

        if shuffle:
            np.random.shuffle(sample_indices)

        sampled_data = [filtered_graphs[index] for index in sample_indices]
        sampled_labels = [filtered_labels[index] for index in sample_indices]

        return sampled_data, sampled_labels

    def sampling_by_count(self, count, target_label=1, graph_augment=False):
        assert (target_label in [0, 1])
        # filtering out the graphs that contain only one node.
        if graph_augment:
            filtered_labels = [graph.y for graph in self.data_list if (graph.num_nodes > 1)]
            filtered_graphs = [graph for graph in self.data_list if (graph.num_nodes > 1)]
            print(f"{len(filtered_labels)}/{len(self.labels)} graphs that contain the specified unique nodes")
        else:
            filtered_labels = self.labels
            filtered_graphs = self.data_list

        # array_labels = self.labels if isinstance(self.labels, np.ndarray) else np.array(self.labels)
        array_labels = filtered_labels if isinstance(filtered_labels, np.ndarray) else np.array(filtered_labels)
        matched_data_indices = np.where(array_labels == target_label)
        if len(matched_data_indices[0]) > count:
            sampled_indices = np.random.choice(matched_data_indices[0], size=count, replace=False)
        else:
            extra_indices = np.random.choice(matched_data_indices[0], size=count - len(matched_data_indices[0]), replace=True)
            sampled_indices = np.concatenate((matched_data_indices[0], extra_indices))

        return sampled_indices, filtered_graphs, filtered_labels

