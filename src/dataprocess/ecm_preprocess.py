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
                "window_type": "session" if data_name == 'hdfs' else "sliding",
                "use_tfidf": w_tfidf,
                "label_type": "anomaly",
                "window_size": "session" if data_name == 'hdfs' else w_size,
                "stride": w_size if s_size is None else s_size,
            }
    return para_config


if __name__ == "__main__":

    root_path = f"../../dataset/processed/"
    for dataset in ['hdfs', 'bgl', 'spirit']:
        for test_ratio in [0.2]:
            if dataset == 'hdfs':
                windows = ['session']
            else:
                windows = [100, 20]
            for window_size in windows:
                data_dir = get_data_directory(dataset, root_path, test_ratio)
                output_path = data_dir

                params = set_param_configuration(data_name=dataset, data_path=data_dir, w_size=window_size)
                embedding = 'tfidf' if params['use_tfidf'] else 'bert'
                print(f"Starting to process dataset={params['dataset']} test_ratio={test_ratio} window_size={params['window_size']} feature={params['feature_type']}")
                print("*" * 90)

                session_train, session_test = load_sessions(data_dir=data_dir)

                print(f"Completed {dataset} session data loading")
                ext = FeatureExtractor(**params)
                session_train = ext.fit_transform(session_train)
                session_test = ext.transform(session_test, datatype="test")

                data_list = []
                label_list = []

                for session_idx, data_dict in enumerate(session_train.values()):
                    data_list.append(data_dict["eventids"])
                    label_list.append(data_dict["window_anomalies"][0])

                with open(f"{data_dir}/{dataset}-{params['window_size']}logs-train.pkl", 'wb') as f:
                    pickle.dump((data_list, label_list), f)

                for session_idx, data_dict in enumerate(session_test.values()):
                    data_list.append(data_dict["eventids"])
                    label_list.append(data_dict["window_anomalies"][0])

                with open(f"{data_dir}/{dataset}-{params['window_size']}logs-test.pkl", 'wb') as f:
                    pickle.dump((data_list, label_list), f)
                print(f"Completed {dataset} ecm data generation")