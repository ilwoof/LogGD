# -*- coding: utf-8 -*-
"""
Created on Tue April 07 17:16:36 2022

@author: Yongzheng Xie
"""

import os.path
import sys
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter, namedtuple

sys.path.append("../../")

from src.deeploglizer.common.preprocess import FeatureExtractor
from src.deeploglizer.common.dataloader import load_sessions, log_dataset

def write_to_csv(file_name, data_content):
    head = 'label' if 'labels' in file_name else ','.join(['graph_idx', 'src', 'dst', 'weight'])
    with open(file_name, 'w') as f:
        f.write(f"{head}\n")
        f.write(f"{data_content}")


def generate_labels(d_name, d_type, w_size, f_type, data_content):
    prefix = f"{w_size}_{f_type}"
    if d_type is None:
        filename = os.path.join(output_path, f'{d_name}_{prefix}_labels.csv')
    else:
        filename = os.path.join(output_path, f'{d_name}_{d_type}_{prefix}_labels.csv')
    file_content = '\n'.join(list(map(str, data_content)))
    write_to_csv(filename, file_content)
    print(f"complete label data processing\n")


def generate_edge_info(d_name, d_type, w_size, f_type, data_content):
    prefix = f"{w_size}_{f_type}"
    if d_type is None:
        filename = os.path.join(output_path, f'{d_name}_{prefix}_edges.csv')
    else:
        filename = os.path.join(output_path, f'{d_name}_{d_type}_{prefix}_edges.csv')
    file_content = '\n'.join(data_content)
    write_to_csv(filename, file_content)
    print(f"complete edge data processing")


def construct_graph_level_data(d_name, d_list, w_size, f_type, args_para, d_type=None, keep_self_loop=False):
    edges_info_list = []
    labels_list = []
    valid_graph = 0
    invalid_graph = 0
    average_sequence_len = 0
    average_graph_unique_nodes = 0
    anomalous_graphs = 0
    removed_anomalous_graphs = 0

    suppose_len = 48 if w_size == 'session' else w_size
    unique_event_statistics = {i+1: 0 for i in range(suppose_len)}
    unique_event_anomaly_statistics = {i+1: 0 for i in range(suppose_len)}

    for idx, window_sequence in enumerate(d_list):
        edges_in_window = []
        sequence = window_sequence['features_sequential']
        average_sequence_len += len(sequence)

        if args_para.use_statistics:
            average_graph_unique_nodes += len(set(sequence))
            unique_event_statistics[len(set(sequence))] += 1
            unique_event_anomaly_statistics[len(set(sequence))] += window_sequence['window_anomalies']

        # for the starting event, a self_loop is added to create an extra edge that will be used in downstream task
        if len(sequence) > 0 and keep_self_loop:
            edges_in_window.append(f"{sequence[0]},{sequence[0]}")
        
        for index in range(len(sequence) - 1):
            source = sequence[index]
            target = sequence[index + 1]
            # eliminate the edges of self-loop
            if source == target and not keep_self_loop:
                continue
            edges_in_window.append(f"{source},{target}")

        for k, v in Counter(edges_in_window).items():
            edge_info = f"{idx},{k},{v}"
            edges_info_list.append(edge_info)

        # if keep_self_loop is False, remove the labels for those graphs that only contain self-loop edges
        if len(edges_in_window) > 0:
            labels_list.append(window_sequence['window_anomalies'])
            if window_sequence['window_anomalies'] == 1:
                anomalous_graphs += 1
            valid_graph += 1
        else:
            # for the removed graph, their labels are also skipped
            invalid_graph += 1
            if window_sequence['window_anomalies'] == 1:
                removed_anomalous_graphs += 1

        if (idx + 1) % 10000 == 0:
            print(f"processing session {idx + 1}", end='\r')

    print(f"Generated {d_type} graphs={valid_graph}({anomalous_graphs} anomalies)")
    print(f"Removed {d_type} graphs={invalid_graph}({removed_anomalous_graphs} anomalies)\n")

    if args_para.use_statistics:
        y = np.fromiter(unique_event_statistics.values(), dtype=float)
        ind = np.argsort(-y)[:10]
        cumulative_percentage = 0
        print(f"Dataset ({d_name}) Graph Analysis:")
        print(f"Average sequence length:{(average_sequence_len/len(d_list)):.3f}")
        print(f"Average unique nodes/graph:{(average_graph_unique_nodes/len(d_list)):.3f}")
        print(f"Unique nodes for top10 graphs and percentage:")

        rank_statistics = {}
        for u in ind:
            node_graph_percentage = y[u]/len(d_list)
            cumulative_percentage += node_graph_percentage
            if node_graph_percentage > 0:
                rank_statistics[f'{u+1} events'] = node_graph_percentage*100
                print(f"    {u+1} Unique Events : {node_graph_percentage*100:.3f}%, "
                      f"Cumulative Percentage={cumulative_percentage*100:.3f}%, "
                      f"Anomalies={unique_event_anomaly_statistics[u+1]}/{y[u]}")
        rank_statistics[f'others'] = 100.0 - cumulative_percentage*100
        print("\n")
        title = f'{d_name.upper()}-{w_size}' if d_name == 'hdfs' else f'{d_name.upper()}-{w_size}logs'

        x = list(rank_statistics.values())
        legend_labels = [f"{key} - {value:.2f}%" for key, value in rank_statistics.items()]

        fig, ax = plt.subplots()
        ax.pie(x)
        ax.set_title(title)
        plt.legend(labels=legend_labels, loc='best')
        # title = f"The Proportion of Sequences by Number of Unique Events"
        if not os.path.exists("./png"):
            os.mkdir("./png")
        plt.savefig(f'./png/{d_name}_{w_size}logs_{d_type}.png')
        plt.clf()

    generate_edge_info(d_name, d_type, w_size, f_type, edges_info_list)
    generate_labels(d_name, d_type, w_size, f_type, labels_list)
    return


def get_data_directory(data_name, root_dir, ratio_set):
    return f"{root_dir}/{data_name}/{data_name.lower()}_{ratio_set}_tar/"

def set_param_configuration(data_name, data_path, w_size='session', s_size=None, s_embedding='bert'):
    para_config = {
                "data_dir": data_path,
                "dataset": data_name,
                "feature_type": "semantics",  # "sequentials", "semantics", "quantitatives"
                "window_type": "session" if data_name == 'hdfs' else "sliding",
                "embedding_type": s_embedding,
                "label_type": "anomaly",
                "window_size": w_size,
                "stride": w_size if s_size is None else s_size
            }
    return para_config


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Graph data generation')
    parser.add_argument('--datasets', nargs="+", default=['hdfs', 'bgl', 'spirit', 'tbd'], help='The dataset to be processed')
    parser.add_argument('--keep_selfloop', default=True, help='Whether consecutively repetitive events are kept in graphs')
    parser.add_argument('--embedding_type', default='bert', choices=['bert', 'tfidf', 'w2v'], help='The method to generate embedding')
    parser.add_argument('--window_size', nargs="+", default=[100, 60, 20], type=int, help='window size')
    parser.add_argument('--use_statistics', action="store_true", help='Whether to output the statistics for duplicate events')
    args = parser.parse_args()

    root_path = f"../../dataset/processed/"
    for dataset in args.datasets:
        for test_ratio in [0.2]:
            if dataset == 'hdfs':
                windows = ['session']
            else:
                windows = args.window_size
            for window_size in windows:
                data_dir = get_data_directory(dataset, root_path, test_ratio)
                output_path = data_dir

                params = set_param_configuration(data_name=dataset, data_path=data_dir,
                                                 w_size=window_size, s_embedding=args.embedding_type
                                                 )
                print(f"Starting to process dataset={params['dataset']} test_ratio={test_ratio} "
                      f"window_size={params['window_size']} embedding_type={params['embedding_type']} keep_selfloop={args.keep_selfloop}")
                print("*" * 100)

                # session_train, session_test = load_sessions(data_dir=data_dir)

                with open(f'{root_path}/{dataset}/{dataset}_{window_size}_train.pkl', mode='rb') as f:
                    session_train = pickle.load(f)

                with open(f'{root_path}/{dataset}/{dataset}_{window_size}_test.pkl', mode='rb') as f:
                    session_test = pickle.load(f)

                print(f"Completed {dataset} session data loading")
                ext = FeatureExtractor(**params)
                session_train = ext.fit_transform(session_train)
                session_test = ext.transform(session_test, datatype="test")

                print(f"Completed {dataset} feature data extraction\n")
                # EventIds is used as the node index
                dataset_train = log_dataset(session_train, feature_type="semantics")
                dataset_test = log_dataset(session_test, feature_type="semantics")

                # with open(f'{root_path}/{dataset}/{dataset}_{window_size}_meta_data.pkl', mode='wb') as f:
                #     pickle.dump(ext.meta_data, f)
                #
                # with open(f'{root_path}/{dataset}/{dataset}_{window_size}_train_features.pkl', mode='wb') as f:
                #     pickle.dump(dataset_train.flatten_data_list, f)
                #
                # with open(f'{root_path}/{dataset}/{dataset}_{window_size}_test_features.pkl', mode='wb') as f:
                #     pickle.dump(dataset_test.flatten_data_list, f)

                print(f"Starting to generate {dataset} graph data")
                construct_graph_level_data(dataset, dataset_train.flatten_data_list,
                                           params['window_size'], params['feature_type'],
                                           args_para=args, d_type='train', keep_self_loop=args.keep_selfloop)
                construct_graph_level_data(dataset, dataset_test.flatten_data_list,
                                           params['window_size'], params['feature_type'],
                                           args_para=args, d_type='test', keep_self_loop=args.keep_selfloop)
