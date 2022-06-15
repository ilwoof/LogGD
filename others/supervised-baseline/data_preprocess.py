# -*- coding: utf-8 -*-
"""
Created on Tue April 07 17:16:36 2022

@author: Yongzheng Xie
"""

import os.path
from collections import Counter
from deeploglizer.common.preprocess import FeatureExtractor
from deeploglizer.common.dataloader import load_sessions, log_dataset

datasets = [{'name': 'HDFS', 'node_num': 18},
            {'name': 'BGL', 'node_num': 122},
            ]

def write_to_csv(file_name, data_content):
    if 'labels' in file_name:
        head = 'label'
    else:
        head_columns = ['graph_idx', 'src', 'dst', 'weight']
        head = ','.join(head_columns)

    with open(file_name, 'w') as f:
        f.write(f"{head}\n")
        f.write(f"{data_content}")

def generate_labels(data_type, data_content):
    if data_type is None:
        filename = os.path.join(output_path, f'{dataset_name}_graph_labels.csv')
    else:
        filename = os.path.join(output_path, f'{dataset_name}_{data_type}_graph_labels.csv')
    file_content = '\n'.join(list(map(str, data_content)))
    write_to_csv(filename, file_content)

def generate_edge_info(data_type, data_content):
    if data_type is None:
        filename = os.path.join(output_path, f'{dataset_name}_edges.csv')
    else:
        filename = os.path.join(output_path, f'{dataset_name}_{data_type}_edges.csv')
    file_content = '\n'.join(data_content)
    write_to_csv(filename, file_content)

def construct_graph_level_data(data_list, data_type=None):
    edges_info_list = []
    labels_list = []
    valid_graph = 0
    invalid_graph = 0

    for idx, window_sequence in enumerate(data_list):
        edges_in_window = []
        sequence = window_sequence['features']

        for index in range(len(sequence) - 1):
            source = sequence[index]
            destination = sequence[index + 1]
            # eliminate the edges of self-loop
            if source == destination:
                continue
            edges_in_window.append(f"{source},{destination}")

        for k, v in Counter(edges_in_window).items():
            edge_info = f"{idx},{k},{v}"
            edges_info_list.append(edge_info)

        # remove the labels for those graphs that only contain self-loop edges
        if len(edges_in_window) > 0:
            labels_list.append(window_sequence['window_anomalies'])
            valid_graph += 1
        else:
            print(f"remove the labels for the empty graphs {idx}")
            invalid_graph += 1

        if (idx+1) % 10000 == 0:
            print(f"processing session {idx + 1}")

    print(f"valid graph={valid_graph}, invalid graph={invalid_graph}")
    generate_edge_info(data_type, edges_info_list)
    generate_labels(data_type, labels_list)
    return


if __name__ == "__main__":

    for dataset in datasets:
        dataset_name = dataset['name']
        node_num = dataset['node_num']
        data_dir = f"dataset/{dataset_name}/{dataset_name.lower()}_0.0_tar"
        output_path = f"./dataset/{dataset_name}"
        if dataset_name == 'HDFS':
            params = {
                "data_dir": data_dir,
                "dataset": dataset_name,
                "feature_type": "semantics",  # "sequentials", "semantics", "quantitatives"
                "window_type": "session",  # "session", "sliding"
                # "use_tfidf": True,
                "label_type": "anomaly",  # "none", "next_log", "anomaly"
            }
        else:
            params = {
                "data_dir": data_dir,
                "dataset": dataset_name,
                "feature_type": "semantics",  # "sequentials", "semantics", "quantitatives"
                "window_type": "sliding",  # "session", "sliding"
                "label_type": "anomaly",  # "none", "next_log", "anomaly"
                # "use_tfidf": True,
                "window_size": 20,
                "stride": 1
            }
        print(f"Starting to process dataset {dataset_name}")
        session_train, session_test = load_sessions(data_dir=data_dir)
        print(f"Completing {dataset_name} session data loading")
        ext = FeatureExtractor(**params)
        session_train = ext.fit_transform(session_train)
        session_test = ext.transform(session_test, datatype="test")
        print(f"Completing {dataset_name} feature data extraction")
        # dataset_train = log_dataset(session_train, feature_type="sequentials")
        dataset_test = log_dataset(session_test, feature_type="sequentials")
        print(f"Starting to generate {dataset_name} graph data")
        #  construct_graph_level_data(dataset_train.flatten_data_list, 'train')
        construct_graph_level_data(dataset_test.flatten_data_list, 'test')
