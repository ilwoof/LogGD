import os
import pickle
import argparse
import pandas as pd
import numpy as np
from utils import decision, json_pretty_dump
from collections import OrderedDict, defaultdict

parser = argparse.ArgumentParser()

parser.add_argument('--DS', dest='DS', help='dataset name')
parser.add_argument("--test_ratio", default=0.2, type=float)

init_params = vars(parser.parse_args())

dataset_name = init_params['DS']
if dataset_name == 'tbd':
    log_file_name = 'Thunderbird10M'
elif dataset_name == 'spirit':
    log_file_name = 'Spirit1G'
else:
    log_file_name = dataset_name.upper()

eval_name = f'{dataset_name}_{init_params["test_ratio"]}_tar'
seed = 42
data_dir = f"../../dataset/processed/{dataset_name}"
np.random.seed(seed)

params = {
    "log_file": f"../../dataset/raw/{dataset_name}/{log_file_name}.log_structured.csv",
    "time_range": 21600,  # 6 hours
    "train_ratio": None,
    "test_ratio": init_params['test_ratio'],
    "random_sessions": False if dataset_name == 'bgl' else True,
    "train_anomaly_ratio": 1.0,
}

data_dir = os.path.join(data_dir, eval_name)
os.makedirs(data_dir, exist_ok=True)


def load_NonHDFS(
    log_file,
    time_range,
    train_ratio,
    test_ratio,
    random_sessions,
    train_anomaly_ratio,
):
    print(f"Loading {dataset_name} logs from {log_file}")
    struct_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)
    # struct_log.sort_values(by=["Timestamp"], inplace=True)

    struct_log["Label"] = struct_log["Label"].map(lambda x: x != "-").astype(int).values
    if 'bgl' in dataset_name:
        struct_log["datetime"] = pd.to_datetime(struct_log['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
    else:
        struct_log['datetime'] = pd.to_datetime(struct_log["Date"] + " " + struct_log['Time'], format='%Y-%m-%d %H:%M:%S')

    # struct_log["seconds_since"] = ((struct_log["time"] - struct_log["time"][0]).dt.total_seconds().astype(int))
    struct_log["seconds_since"] = ((struct_log['datetime'] - struct_log["datetime"][0]).dt.total_seconds().astype(int))
    struct_log["seconds_since"].fillna(0)

    session_dict = OrderedDict()
    column_idx = {col: idx for idx, col in enumerate(struct_log.columns)}
    for idx, row in enumerate(struct_log.values):
        current = row[column_idx["seconds_since"]]
        if idx == 0:
            sessid = current
        elif current - sessid > time_range:
            sessid = current
        if sessid not in session_dict:
            session_dict[sessid] = defaultdict(list)
        session_dict[sessid]["eventids"].append(row[column_idx["EventId"]])
        session_dict[sessid]["templates"].append(row[column_idx["EventTemplate"]])
        session_dict[sessid]["label"].append(row[column_idx["Label"]]
        )  # labeling for each log

    # labeling for each session
    # for k, v in session_dict.items():
    #     session_dict[k]["label"] = [int(1 in v["label"])]

    session_idx = list(range(len(session_dict)))
    # split data
    if random_sessions:
        print("Using random partition.")
        np.random.shuffle(session_idx)

    session_ids = np.array(list(session_dict.keys()))

    if train_ratio is None:
        train_ratio = 1 - test_ratio
    train_lines = int(train_ratio * len(session_idx))
    test_lines = int(test_ratio * len(session_idx))

    session_idx_train = session_idx[0:train_lines]
    session_idx_test = session_idx[-test_lines:]

    session_id_train = session_ids[session_idx_train]
    session_id_test = session_ids[session_idx_test]

    print("Total # sessions: {}".format(len(session_ids)))

    session_train = {
        k: session_dict[k]
        for k in session_id_train
        if (sum(session_dict[k]["label"]) == 0)
        or (sum(session_dict[k]["label"]) > 0 and decision(train_anomaly_ratio))
    }
    session_test = {k: session_dict[k] for k in session_id_test}

    session_labels_train = [
        1 if sum(v["label"]) > 0 else 0 for _, v in session_train.items()
    ]
    session_labels_test = [
        1 if sum(v["label"]) > 0 else 0 for _, v in session_test.items()
    ]

    train_anomaly = 100 * sum(session_labels_train) / len(session_labels_train) if len(session_labels_train) > 0 else 0
    test_anomaly = 100 * sum(session_labels_test) / len(session_labels_test) if len(session_labels_test) > 0 else 0

    print("# train sessions: {} ({:.2f}%)".format(len(session_train), train_anomaly))
    print("# test sessions: {} ({:.2f}%)".format(len(session_test), test_anomaly))

    with open(os.path.join(data_dir, "session_train.pkl"), "wb") as fw:
        pickle.dump(session_train, fw)
    with open(os.path.join(data_dir, "session_test.pkl"), "wb") as fw:
        pickle.dump(session_test, fw)
    json_pretty_dump(params, os.path.join(data_dir, "data_desc.json"))
    print("Saved to {}".format(data_dir))
    return session_train, session_test


if __name__ == "__main__":
    load_NonHDFS(**params)
