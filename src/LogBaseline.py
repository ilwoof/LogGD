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
import logging
from numpy.random import seed
from collections import OrderedDict

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

from tansform_new import ShortestPathGenerator, LogCollator

from evaluate import record_result, format_output, gather_results

from model.gcn_model import GCNNetwork
from model.gin_model import GINNetwork
from model.gat_model import GATNetwork
from model.gtc_model import GTCNetwork

from model.lr import PolynomialDecayLR
from early_stopping import EarlyStopping

np.set_printoptions(threshold=np.inf)
model_path = f"../result/model/"
result_path = f"../result/log/"
result_file_name = f"_experiment_results.csv"
model_file_name = f"_checkpoint.pt"

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def arg_parse():
    parser = argparse.ArgumentParser(description='LogGT Arguments')
    parser.add_argument('--data_dir', dest='data_dir', default='../dataset/processed/',
                        help='Directory where data is located')
    parser.add_argument('--model', default="gat", choices=['gin', 'gine', 'gcn2', 'dgcn', 'gat', 'gtc'],
                        help='The model to be used')
    parser.add_argument('--datasets', nargs="+", default=['bgl', 'hdfs', 'spirit', 'tbd'],
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
    parser.add_argument("--max_hop", type=int, default=80)
    parser.add_argument('--num_layer', dest='num_layer', default=4, type=int, help='Encoder layer number')
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--perturb_noise", default=0.00, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--peak_lr", default=3e-4, type=float)
    parser.add_argument("--end_lr", default=1e-9, type=float)
    parser.add_argument("--warmup_epoch", default=3, type=int)
    parser.add_argument('--dropout', dest='dropout', default=0.3, type=float, help='Dropout rate')
    parser.add_argument('--patience', dest='patience', default=20, type=int, help='Early stopping after patience')
    parser.add_argument('--seed', dest='seed', type=int, default=42, help='seed')
    parser.add_argument('--debug_mode', action="store_true", help='The flag is used to indicate one round test')
    parser.add_argument('--no_validation', action="store_true", help='Whether to use validation set for the results')
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

            optimizer.zero_grad()
            batch_data.requires_grad_(False)
            output = model(batch_data.to(device))
            labels = batch_data.y.to(device)

            loss = criterion(output, labels)
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
                    output = model(batch_data.to(device))
                    labels = batch_data.y.to(device)

                    loss = criterion(output, labels)
                    valid_pred.append(loss.cpu().detach().numpy().flatten())

            valid_loss = np.array(valid_pred).mean()

            elapsed_time = time.time() - begin_time
            print(f'[{epoch + 1}/{args.num_epochs}] train_loss: {train_loss:.7f} valid_loss: {valid_loss:.7f} elapsed time:{elapsed_time:.5f}')

            if not early_stopping(valid_loss, model, monitor_metric='loss'):
                continue
            else:
                # load the last checkpoint with the best model
                model.load_state_dict(torch.load(model_file))

        ######################
        # test the model #
        ######################
        model.eval()
        with torch.no_grad():
            batch_loss = []
            test_pred = []
            y = []
            data_loader = iter(test_loader)
            for batch_idx, batch_data in enumerate(tqdm(data_loader, desc="testing")):
                test_begin_time = time.time()

                output = model(batch_data.to(device))
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

            precision, recall, f1_score, _ = precision_recall_fscore_support(ground_truth, np.argmax(pred_label, axis=1),
                                                                             average='binary')
            results = {'loss': np.array(batch_loss).mean(), 'precision': precision, 'recall': recall, 'f1_score': f1_score, 'roc': test_roc_ab}

        # if validation set is used, directly report the results for test set
        if valid_loader is not None:
            best_metric = results
            break
        else:
            print(f"[{epoch + 1}/{args.num_epochs}] precision={precision:.5f}, recall={recall:.5f}, f1-score={f1_score:.5f}, auc-roc={test_roc_ab:.5f}\n")
            # if validation set is None, use the best f1_score results for test set
            if early_stopping(results, model, monitor_metric='f1_score') or (epoch + 1) == args.num_epochs:
                best_metric = early_stopping.get_best_result()
                break
            else:
                continue

    result_details = OrderedDict([
            ('precision', f"{best_metric['precision']:.5f}"),
            ('recall', f"{best_metric['recall']:.5f}"),
            ('f1-score', f"{best_metric['f1_score']:.5f}"),
            ('auc-roc', f"{best_metric['roc']:.5f}"),
            (f'training time({epoch + 1} epochs)', f"{np.array(train_time_list).sum():.5f}s"),
            ('testing time(per batch)', f"{np.array(test_time_list).mean() * 1000:.5f}(ms)"),

        ])

    logging.info(format_output(result_details))

    return best_metric


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

                # use generator to save memories
                te_size = len(graphs_test)
                te_anomaly_num = sum(graphs_test.get_labels().numpy())
                te_anomaly_ratio = float(te_anomaly_num/te_size)

                for sample_num in args.sampling_training:
                    for anomaly_ratio in args.anomaly_ratio:
                        training_data, training_labels = tr_graphs.get_samples(sample_size=sample_num,
                                                                               anomaly_ratio=anomaly_ratio)

                        data_setting = OrderedDict([
                                        ('model_name', args.model),
                                        ('data_set', data_set),
                                        ('window_size', window_size),
                                        ('embedding_type', args.embedding_type),
                                        ('epochs', args.num_epochs),
                                        ('patience', args.patience),
                                        ('no_validation', args.no_validation),
                                        ('max_hops', args.max_hop),
                                        ('test_ratio', test_ratio),
                                        ('sampling_ratio', sample_num),
                                        ('anomaly_ratio', anomaly_ratio),
                                        ('dataset_nodes', tr_graphs.dataset_node_size),
                                        ('max_graph_nodes', tr_graphs.max_graph_node_size),
                                        ])

                        logging.info(format_output(data_setting))

                        kfd = StratifiedShuffleSplit(n_splits=3, train_size=int(len(training_labels) * 0.9),
                                                     random_state=args.seed)

                        result_dict = {'roc': [], 'precision': [], 'recall': [], 'f1': []}

                        for tr_index, va_index in kfd.split(training_data, training_labels):

                            if args.no_validation:
                                graphs_train = training_data
                                graphs_valid = []
                                tr_size = len(graphs_train)
                                va_size = len(graphs_valid)
                                tr_anomaly_num = sum(1 for graph in graphs_train if graph.y == 1)
                                va_anomaly_num = 0
                                tr_anomaly_ratio = float(tr_anomaly_num / tr_size)
                                va_anomaly_ratio = 0.0

                            else:
                                graphs_train = [training_data[idx] for idx in tr_index]
                                graphs_valid = [training_data[idx] for idx in va_index]

                                # use generator to save memories
                                tr_size = len(graphs_train)
                                va_size = len(graphs_valid)
                                tr_anomaly_num = sum(1 for graph in graphs_train if graph.y == 1)
                                va_anomaly_num = sum(1 for graph in graphs_valid if graph.y == 1)
                                tr_anomaly_ratio = float(tr_anomaly_num / tr_size)
                                va_anomaly_ratio = float(va_anomaly_num / va_size)

                            data_split_details = OrderedDict([
                                ('training(size/anomalies/ratio)', f"{tr_size}/{tr_anomaly_num}/{tr_anomaly_ratio:.3f}"),
                                ('validation', f"{va_size}/{va_anomaly_num}/{va_anomaly_ratio:.3f}"),
                                ('test', f"{te_size}/{te_anomaly_num}/{te_anomaly_ratio:.3f}"),
                                ])

                            logging.info(format_output(data_split_details))

                            if args.model == 'gcn2':
                                model = GCNNetwork(out_dim=args.output_dim,
                                                   d_model=args.embedding_dim,
                                                   num_layer=args.num_layer,
                                                   dropout=args.dropout,
                                                   num_node_type=-node_feat_dim,
                                                   perturb_noise=args.perturb_noise,
                                                   gnn_type='gcn2',
                                                   ).to(device)

                            elif args.model == 'dgcn':
                                model = GCNNetwork(out_dim=args.output_dim,
                                                   d_model=args.embedding_dim,
                                                   num_layer=args.num_layer,
                                                   dropout=args.dropout,
                                                   num_node_type=-node_feat_dim,
                                                   perturb_noise=args.perturb_noise,
                                                   gnn_type='dgcn',
                                                   ).to(device)

                            elif args.model == 'gin':
                                model = GINNetwork(out_dim=args.output_dim,
                                                   d_model=args.embedding_dim,
                                                   num_layer=args.num_layer,
                                                   num_node_type=-node_feat_dim,
                                                   perturb_noise=args.perturb_noise,
                                                   dropout=args.dropout,
                                                   gnn_type='gin'
                                                   ).to(device)

                            elif args.model == 'gine':
                                model = GINNetwork(out_dim=args.output_dim,
                                                   d_model=args.embedding_dim,
                                                   num_layer=args.num_layer,
                                                   num_node_type=-node_feat_dim,
                                                   perturb_noise=args.perturb_noise, dropout=args.dropout,
                                                   gnn_type='gine'
                                                   ).to(device)

                            elif args.model == 'gat':
                                model = GATNetwork(out_dim=args.output_dim,
                                                   d_model=args.embedding_dim,
                                                   num_layer=2,
                                                   nhead=args.nhead,
                                                   dropout=args.dropout,
                                                   attention_dropout=args.dropout,
                                                   num_node_type=-node_feat_dim,
                                                   perturb_noise=args.perturb_noise,
                                                   ).to(device)

                            elif args.model == 'gtc':
                                model = GTCNetwork(out_dim=args.output_dim,
                                                   d_model=args.embedding_dim,
                                                   num_layer=2,
                                                   nhead=args.nhead,
                                                   dropout=args.dropout,
                                                   num_node_type=-node_feat_dim,
                                                   perturb_noise=args.perturb_noise,
                                                   ).to(device)
                            else:
                                raise ValueError(f"The specified model {args.model} is not supported")

                            data_train_loader = DataLoader(graphs_train,
                                                           shuffle=True,
                                                           batch_size=args.batch_size,
                                                           collate_fn=Compose([LogCollator()]),
                                                           )

                            if args.no_validation:
                                data_valid_loader = None
                            else:
                                assert len(graphs_valid) > 0
                                data_valid_loader = DataLoader(graphs_valid,
                                                               shuffle=True,
                                                               batch_size=args.batch_size,
                                                               collate_fn=Compose([LogCollator()]))

                            data_test_loader = DataLoader(graphs_test,
                                                          shuffle=False,
                                                          batch_size=args.batch_size,
                                                          collate_fn=Compose([LogCollator()]),
                                                          )

                            one_round_result = train(data_train_loader, data_valid_loader, data_test_loader, model, args)

                            result_dict = gather_results(result_dict, one_round_result)

                            del data_train_loader, data_valid_loader, data_test_loader
                            del graphs_train, graphs_valid

                            if args.debug_mode:
                                print(f"****** To save time, only one round test is conducted in debug mode ******\n")
                                break

                        del training_data, training_labels

                        record_result(result_file, data_setting, result_dict)
