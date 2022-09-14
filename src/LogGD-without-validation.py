# -*- coding: utf-8 -*-
"""
Created on Tue April 07 17:16:36 2022

@author: Yongzheng Xie
"""
import os
import time
import argparse
import numpy as np
import random
from numpy.random import seed

import torch
import torch.nn as nn
import torch.backends.cudnn
from torch.utils.data import DataLoader

from torchvision.transforms import Compose

from sklearn.metrics import auc, roc_curve, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from feature_extraction import GraphFeatureExtractor
from configuration import device
from tansform import ShortestPathGenerator, LogCollator
from evaluate import record_result, log_info
from model.rpgt_model_v9_baseline_modified import GRPENetwork
from model.lr import PolynomialDecayLR
from early_stopping import EarlyStopping

np.set_printoptions(threshold=np.inf)
model_path = f"../result/model/"
result_path = f"../result/log/"
result_file_name = f"_experiment_results.txt"
model_file_name = f"_checkpoint.pt"


def arg_parse():
    parser = argparse.ArgumentParser(description='LogGT Arguments')
    parser.add_argument('--data_dir', dest='data_dir', default='../dataset/processed/',
                        help='Directory where benchmark is located')
    parser.add_argument('--model', default="LogGT", help='The model to be used')
    parser.add_argument('--datasets', nargs="+", default=['bgl', 'hdfs', 'tbd', 'spirit'],
                        help='The dataset to be processed')
    parser.add_argument("--sampling_training", nargs="+", default=[1.0], type=float, help='Train data sampling')
    parser.add_argument('--feature_type', dest='feature_type', default='semantics', help='use what node feature')
    parser.add_argument('--embedding_type', dest='embedding_type', choices=['tfidf', 'bert'], default='bert', type=str,
                        help='tfidf or bert is used')
    parser.add_argument("--window_size", nargs="+", default=[100, 80, 60, 40, 20], type=int, help='window size')
    parser.add_argument("--anomaly_ratio", nargs="+", default=[1.0], type=float)
    parser.add_argument('--clip', dest='clip', default=5.0, type=float, help='Gradient clipping')
    parser.add_argument('--num_epochs', dest='num_epochs', default=100, type=int, help='total epoch number')
    parser.add_argument('--batch_size', dest='batch_size', default=64, type=int, help='Batch size')
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument('--ffn_dim', dest='ffn_dim', default=2048, type=int, help='Feed forward network dimension')
    parser.add_argument('--output_dim', dest='output_dim', default=2, type=int, help='Output dimension')
    parser.add_argument("--edge_weight", type=int, default=1, help='The max number of edge weight')
    parser.add_argument("--max_hop", type=int, default=80)
    parser.add_argument('--num_layer', dest='num_layer', default=1, type=int, help='Encoder layer number')
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--perturb_noise", default=0.00, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--peak_lr", default=3e-4, type=float)
    parser.add_argument("--end_lr", default=1e-9, type=float)
    parser.add_argument("--warmup_epoch", default=3, type=int)
    parser.add_argument('--dropout', dest='dropout', default=0.3, type=float, help='Dropout rate')
    parser.add_argument('--patience', dest='patience', default=10, type=int, help='Early stopping after patience')
    parser.add_argument('--seed', dest='seed', type=int, default=42, help='seed')
    parser.add_argument('--debug_mode', action="store_true", help='The flag is used to indicate one round test')
    return parser.parse_args()


def train(train_loader, valid_loader, test_loader, model, args):
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.peak_lr, weight_decay=args.weight_decay)
    scheduler = PolynomialDecayLR(optimizer, warmup_updates=args.warmup_epoch * len(train_loader),
                                  tot_updates=args.num_epochs * len(train_loader), lr=args.peak_lr,
                                  end_lr=args.end_lr, power=1.0)

    train_time_list = []
    test_time_list = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=args.patience, delta=1e-6, verbose=False, path=model_file)

    for epoch in range(args.num_epochs):

        ######################
        # training the model #
        ######################
        train_pred_loss = []
        model.train()
        criterion = nn.CrossEntropyLoss()

        begin_time = time.time()
        data_loader = iter(train_loader)
        for batch_idx, batch_data in enumerate(data_loader):
            train_begin_time = time.time()
            batch_data = batch_data.to(device)
            batch_data.requires_grad_(False)

            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_data.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            train_pred_loss.append(loss.cpu().detach().numpy().flatten())
            elapsed = time.time() - train_begin_time
            train_time_list.append(elapsed)

        train_loss = np.array(train_pred_loss).mean()

        ######################
        # validate the model #
        ######################
        if valid_loader is not None:
            model.eval()
            with torch.no_grad():
                valid_pred = []
                data_loader = iter(valid_loader)
                for batch_idx, batch_data in enumerate(data_loader):
                    batch_data = batch_data.to(device)
                    output = model(batch_data)
                    loss = criterion(output, batch_data.y)
                    valid_pred.append(loss.cpu().detach().numpy().flatten())

            valid_loss = np.array(valid_pred).mean()
            elapsed_time = time.time() - begin_time
            print(f'[{epoch + 1}/{args.num_epochs}] '
                  f'train_loss: {train_loss:.7f} valid_loss: {valid_loss:.7f} elapsed time:{elapsed_time:.5f}')

            if not early_stopping(valid_loss, model, monitor_metric='loss'):
                continue

        ######################
        # test the model #
        ######################
        if valid_loader is not None:
            # load the last checkpoint with the best model
            model.load_state_dict(torch.load(model_file))

        model.eval()
        with torch.no_grad():
            batch_loss = []
            test_pred = []
            y = []
            data_loader = iter(test_loader)
            for batch_idx, batch_data in enumerate(tqdm(data_loader, desc="testing")):
                test_begin_time = time.time()
                batch_data = batch_data.to(device)
                output = model(batch_data)
                loss = criterion(output, batch_data.y)
                batch_loss.append(loss.cpu().detach().numpy().flatten())

                output = output.softmax(dim=-1)
                test_pred.extend(output.cpu().detach().numpy())
                y.extend(batch_data.y.cpu().detach().numpy().flatten())

                elapsed = time.time() - test_begin_time
                test_time_list.append(elapsed)

            ground_truth = np.array(y)
            pred_label = np.array(test_pred)

            fpr_ab, tpr_ab, _ = roc_curve(ground_truth, np.max(pred_label, axis=1))
            test_roc_ab = auc(fpr_ab, tpr_ab)

            precision, recall, F1_score, _ = precision_recall_fscore_support(ground_truth, np.argmax(pred_label, axis=1), average='binary')

            results = {'loss': np.array(batch_loss).mean(), 'precision': precision, 'recall': recall, 'f1_score': F1_score, 'roc': test_roc_ab}

        if valid_loader is None:
            print(f"[{epoch + 1}/{args.num_epochs}] precision={precision:.5f}, recall={recall:.5f}, f1-score={F1_score:.5f}, auc-roc={test_roc_ab:.5f}\n")
            if early_stopping(results, model, monitor_metric='f1_score') or (epoch + 1) == args.num_epochs:
                best_metric = early_stopping.get_best_result()
                break
        else:
            best_metric = results
            break

    log_content = f"precision={best_metric['precision']:.5f}, recall={best_metric['recall']:.5f}, " \
                  f"f1-score={best_metric['f1_score']:.5f}, auc-roc={best_metric['roc']:.5f}\n"
    log_info(result_file, log_content, sep_flag='-')

    return best_metric['roc'], best_metric['precision'], best_metric['recall'], best_metric['f1_score']


def get_data_directory(data_name, data_path, ratio_set):
    return f"{data_path}/{data_name}/{data_name.lower()}_{ratio_set}_tar/"


def set_param_configuration(data_name, data_directory, feature_type='semantics', w_size='session', s_size=None,
                            embedding_type='bert'):
    para_config = {
        "data_dir": data_directory,
        "dataset": data_name,
        "feature_type": feature_type,  # "semantics", "non_semantics"
        "window_type": "session" if data_name == 'hdfs' else "sliding",
        "embedding_type": embedding_type,
        "label_type": "anomaly",
        "window_size": w_size,
        "stride": w_size if s_size is None else s_size,
    }
    return para_config


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    args = arg_parse()
    setup_seed(args.seed)

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    model_file = f'{model_path}/{args.model}{model_file_name}'
    result_file = f'{result_path}/{args.model}{result_file_name}'

    for data_set in args.datasets:
        for test_ratio in [0.2]:
            if data_set == 'hdfs':
                windows = ['session']
            else:
                windows = args.window_size
            for window_size in windows:

                data_dir = get_data_directory(data_set, args.data_dir, test_ratio)
                params = set_param_configuration(data_set, data_dir, args.feature_type, window_size,
                                                 embedding_type=args.embedding_type)
                print(f"Starting to process dataset={params['dataset']} window_size={params['window_size']}")

                tr_graphs = GraphFeatureExtractor(root=data_dir,
                                                  data_name=params['dataset'],
                                                  embedding_type=params['embedding_type'],
                                                  dataset_type='train',
                                                  window_size=params['window_size'],
                                                  feature_type=params['feature_type'],
                                                  transform=Compose([ShortestPathGenerator()]),
                                                  args=args)

                ts_graphs = GraphFeatureExtractor(root=data_dir,
                                                  data_name=params['dataset'],
                                                  embedding_type=params['embedding_type'],
                                                  dataset_type='test',
                                                  window_size=params['window_size'],
                                                  feature_type=params['feature_type'],
                                                  transform=Compose([ShortestPathGenerator()]),
                                                  args=args)

                node_feat_dim = tr_graphs.feat_dim
                graphs_test = ts_graphs
                # graphs_test, labels_test = ts_graphs.get_samples(sample_size=1.0, anomaly_ratio=1.0, graph_augment=True)
                # te_anomaly_num = sum(labels_test)

                # use generator to save memories
                te_anomaly_num = sum(graphs_test.get_labels().numpy())

                for sample_num in args.sampling_training:

                    for anomaly_ratio in args.anomaly_ratio:

                        training_data, training_labels = tr_graphs.get_samples(sample_size=sample_num,
                                                                               anomaly_ratio=anomaly_ratio)  # , graph_augment=True)

                        log_content = f"Model={args.model}, Dataset={params['dataset']}, Window_size={params['window_size']}, Embedding_type={args.embedding_type}, " \
                                      f"Epochs={args.num_epochs}, Patience={args.patience}, Max_hops={args.max_hop}\n" \
                                      f"Test_ratio={test_ratio}, Sampling_ratio={sample_num}, Anomaly_ratio={anomaly_ratio}, " \
                                      f"Dataset_nodes={tr_graphs.dataset_node_size}, Max_graph_nodes={tr_graphs.max_graph_node_size}"

                        log_info(result_file, log_content, sep_flag='*')

                        result_dict = {'auc': [], 'precision': [], 'recall': [], 'f1': []}

                        kfd = StratifiedShuffleSplit(n_splits=3, train_size=int(len(training_labels) * 0.9),
                                                     random_state=args.seed)

                        for tr_index, va_index in kfd.split(training_data, training_labels):

                            graphs_train = [training_data[idx] for idx in tr_index]
                            graphs_valid = [training_data[idx] for idx in va_index]

                            # use generator to save memories
                            tr_anomaly_num = sum(1 for graph in graphs_train if graph.y == 1)
                            va_anomaly_num = sum(1 for graph in graphs_valid if graph.y == 1)

                            log_content = f"Training(size/anomalies/ratio)={len(graphs_train)}/{tr_anomaly_num}/{(tr_anomaly_num / len(graphs_train)):.3f}, " \
                                          f"Validation={len(graphs_valid)}/{va_anomaly_num}/{(va_anomaly_num / len(graphs_valid)):.3f}, " \
                                          f"Test={len(graphs_test)}/{te_anomaly_num}/{(te_anomaly_num / len(graphs_test)):.3f}\n"

                            log_info(result_file, log_content, sep_flag=None)

                            model = GRPENetwork(out_dim=args.output_dim, d_model=args.embedding_dim,
                                                dim_feedforward=args.ffn_dim,
                                                num_layer=args.num_layer, nhead=args.nhead, max_hop=args.max_hop,
                                                num_node_type=-node_feat_dim,
                                                num_edge_type=args.edge_weight,
                                                # use edge weight to denote different edge type
                                                perturb_noise=args.perturb_noise, dropout=args.dropout,
                                                ).to(device)

                            data_train_loader = DataLoader(graphs_train,
                                                           shuffle=True,
                                                           batch_size=args.batch_size,
                                                           collate_fn=Compose([LogCollator()]))

                            data_valid_loader = DataLoader(graphs_valid,
                                                           shuffle=True,
                                                           batch_size=args.batch_size,
                                                           collate_fn=Compose([LogCollator()]))

                            data_test_loader = DataLoader(graphs_test,
                                                          shuffle=False,
                                                          batch_size=args.batch_size,
                                                          collate_fn=Compose([LogCollator()]))

                            result = train(data_train_loader, data_valid_loader, data_test_loader, model, args)

                            result_dict['auc'].append(result[0])
                            result_dict['precision'].append(result[1])
                            result_dict['recall'].append(result[2])
                            result_dict['f1'].append(result[3])

                            del data_train_loader, data_valid_loader, data_test_loader
                            del graphs_train, graphs_valid

                            if args.debug_mode:
                                print(f"****** To save time, only one round test is conducted in debug mode ******\n")
                                break

                        del training_data, training_labels

                        record_result(result_file, data_set, window_size, result_dict)
