#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 4/05/2022 11:18 am

@author : Yongzheng Xie
@email : ilwoof@gmail.com
"""
import os.path
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter, namedtuple, defaultdict

sys.path.append("../../")

from src.deeploglizer.common.preprocess import FeatureExtractor
from src.deeploglizer.common.dataloader import load_sessions, log_dataset


def get_data_directory(data_name, root_dir, ratio_set):
    return f"{root_dir}/{data_name}/{data_name.lower()}_{ratio_set}_tar/"

def set_param_configuration(data_name, data_path, w_size='session', s_size=None, w_tfidf=False):
    para_config = {
                "data_dir": data_path,
                "dataset": data_name,
                "feature_type": "sequentials",  # "sequentials", "semantics", "quantitatives"
                "window_type": "sliding",
                "use_tfidf": w_tfidf,
                "label_type": "next_log",
                "window_size": w_size,
                "stride": w_size if s_size is None else s_size,
            }
    return para_config


if __name__ == "__main__":

    root_path = f"../../dataset/processed/"
    for dataset in ['hdfs', 'bgl', 'spirit']:
        for test_ratio in [0.2]:  # , 0.3, 0.4, 0.5]:
            if dataset == 'hdfs':
                windows = ['session']
            else:
                windows = [100, 20]
            for window_size in windows:
                data_dir = get_data_directory(dataset, root_path, test_ratio)
                output_path = data_dir

                params = set_param_configuration(data_name=dataset, data_path=data_dir, w_size=window_size)
                print(f"Starting to process dataset={params['dataset']} test_ratio={test_ratio} window_size={params['window_size']} use_tfidf={params['use_tfidf']}")
                print("*" * 90)

                session_train, session_test = load_sessions(data_dir=data_dir)

                print(f"Completed {dataset} session data loading")
                ext = FeatureExtractor(**params)
                session_train = ext.fit_transform(session_train)
                session_test = ext.transform(session_test, datatype="test")

                dataset_train = log_dataset(session_train, feature_type="sequentials")
                dataset_test = log_dataset(session_test, feature_type="sequentials")

                session_idx = []
                window_labels = []
                window_anomalies = []
                features = []

                for i, data in enumerate(dataset_train.flatten_data_list):
                    session_idx.append(data["session_idx"])
                    window_labels.append(data["window_labels"])
                    window_anomalies.append(data["window_anomalies"])
                    features.append(data["features"])

                data_dict = {"SessionId": np.array(session_idx),
                             "window_y": np.array(window_labels),
                             "y": np.array(window_anomalies),
                             "x": np.array(features)}

                with open(f"{data_dir}/{dataset}-{params['window_size']}logs-train-deeplog.pkl", 'wb') as f:
                    pickle.dump(data_dict, f)

                for i, data in enumerate(dataset_test.flatten_data_list):
                    session_idx.append(data["session_idx"])
                    window_labels.append(data["window_labels"])
                    window_anomalies.append(data["window_anomalies"])
                    features.append(data["features"])

                with open(f"{data_dir}/{dataset}-{params['window_size']}logs-test-deeplog.pkl", 'wb') as f:
                    pickle.dump(data_dict, f)
