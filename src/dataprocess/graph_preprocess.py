# -*- coding: utf-8 -*-
"""
Created on Tue April 07 17:16:36 2022

@author: Yongzheng Xie
"""

import os.path
import sys
import argparse
import pickle
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


def generate_edge_info(d_name, d_type, w_size, f_type, data_content):
    prefix = f"{w_size}_{f_type}"
    if d_type is None:
        filename = os.path.join(output_path, f'{d_name}_{prefix}_edges.csv')
    else:
        filename = os.path.join(output_path, f'{d_name}_{d_type}_{prefix}_edges.csv')
    file_content = '\n'.join(data_content)
    write_to_csv(filename, file_content)


def construct_graph_level_data(d_name, d_list, w_size, f_type, d_type=None, keep_self_loop=False):
    edges_info_list = []
    labels_list = []
    valid_graph = 0
    invalid_graph = 0
    anomalous_graphs = 0
    removed_anomalous_graphs = 0

    for idx, window_sequence in enumerate(d_list):
        edges_in_window = []
        sequence = window_sequence['features']

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
            # print(f"remove the labels for the empty graphs {idx}")
            invalid_graph += 1
            if window_sequence['window_anomalies'] == 1:
                removed_anomalous_graphs += 1

        if (idx + 1) % 10000 == 0:
            print(f"processing session {idx + 1}", end='\r')

    print(f"Valid graph in {d_type}={valid_graph}, {anomalous_graphs} graphs are anomalies"
          f"invalid graph in {d_type}={invalid_graph}, {removed_anomalous_graphs} graphs are anomalies")
    generate_edge_info(d_name, d_type, w_size, f_type, edges_info_list)
    generate_labels(d_name, d_type, w_size, f_type, labels_list)
    return


def get_data_directory(data_name, root_dir, ratio_set):
    return f"{root_dir}/{data_name}/{data_name.lower()}_{ratio_set}_tar/"

def set_param_configuration(data_name, data_path, w_size='session', s_size=None, b_tfidf=False):
    para_config = {
                "data_dir": data_path,
                "dataset": data_name,
                "feature_type": "semantics",  # "sequentials", "semantics", "quantitatives"
                "window_type": "session" if data_name == 'hdfs' else "sliding",
                "use_tfidf": b_tfidf,
                "label_type": "anomaly",
                "window_size": w_size,
                "stride": w_size if s_size is None else s_size
            }
    return para_config


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Graph data generation')
    parser.add_argument('--anomaly_ratio', default=1.0, type=float, help='The dataset with anomaly ratio')
    parser.add_argument('--DS', nargs="+", default=['hdfs', 'bgl', 'spirit', 'tbd'], help='The dataset to be processed')
    parser.add_argument('--keep_selfloop', default=True, action="store_true", help='Whether consecutively repetitive events are kept in graphs')
    parser.add_argument('--use_tfidf', default=False, help='If true use tfidf else use bert')
    args = parser.parse_args()

    root_path = f"../../dataset/processed/"
    for dataset in args.DS:
        for test_ratio in [0.2]:
            if dataset == 'hdfs':
                windows = ['session']
            else:
                windows = [100, 20]
            for window_size in windows:
                data_dir = get_data_directory(dataset, root_path, test_ratio)
                output_path = data_dir

                params = set_param_configuration(data_name=dataset, data_path=data_dir,
                                                 w_size=window_size, b_tfidf=args.use_tfidf
                                                 )
                print(f"Starting to process dataset={params['dataset']} test_ratio={test_ratio} "
                      f"window_size={params['window_size']} embedding={'tfidf'if params['use_tfidf'] else 'bert'}")
                print("*" * 80)

                # session_train, session_test = load_sessions(data_dir=data_dir)

                with open(f'{root_path}/{dataset}/{dataset}_{window_size}_train.pkl', mode='rb') as f:
                    session_train = pickle.load(f)

                with open(f'{root_path}/{dataset}/{dataset}_{window_size}_test.pkl', mode='rb') as f:
                    session_test = pickle.load(f)

                print(f"Completed {dataset} session data loading")
                ext = FeatureExtractor(**params)
                session_train = ext.fit_transform(session_train)
                session_test = ext.transform(session_test, datatype="test")

                print(f"Completed {dataset} feature data extraction")
                # Event Id is used as the node index
                dataset_train = log_dataset(session_train, feature_type="sequentials")
                dataset_test = log_dataset(session_test, feature_type="sequentials")

                # print(f"Starting to generate {dataset} graph data")
                construct_graph_level_data(dataset, dataset_train.flatten_data_list,
                                           params['window_size'], params['feature_type'], 'train', keep_self_loop=args.keep_selfloop)
                construct_graph_level_data(dataset, dataset_test.flatten_data_list,
                                           params['window_size'], params['feature_type'], 'test', keep_self_loop=args.keep_selfloop)
                print("\n")
