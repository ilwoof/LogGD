# -*- coding: utf-8 -*-
"""
Created on Tue April 07 17:16:36 2022

@author: Yongzheng Xie
"""

import time
import argparse
import numpy as np
import random
from numpy.random import seed

import torch
import torch.nn as nn
import torch.backends.cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.transforms import Compose

from sklearn.metrics import auc, precision_recall_curve, roc_curve, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from LogGraphDataset import LogGraphDataset
from feature_extraction import GraphFeatureExtractor, device
from tansform import ShortestPathGenerator, OneHotEdgeAttr, LogCollator
from evaluate import record_result
from model.rpgt_model import GRPENetwork
from model.lr import PolynomialDecayLR
from early_stopping import EarlyStopping

np.set_printoptions(threshold=np.inf)

def arg_parse():
    parser = argparse.ArgumentParser(description='LogGT Arguments')
    parser.add_argument('--data_dir', dest='data_dir', default='../dataset/processed/',
                        help='Directory where benchmark is located')
    parser.add_argument('--model', default="LogGT", help='The model to be used')
    parser.add_argument('--datasets', nargs="+", default=['hdfs', 'bgl', 'spirit'], help='The dataset to be processed')
    parser.add_argument("--sampling_training", nargs="+", default=[6000, 5000, 4000, 3000, 2000, 1000], type=float, help='Train data sampling')
    parser.add_argument('--feature_type', dest='feature_type', default='semantics', help='use what node feature')
    parser.add_argument('--embedding_type', dest='embedding_type', choices=['tfidf', 'bert'], default='bert', type=str, help='tfidf or bert is used')
    parser.add_argument("--window_size", nargs="+", default=[100, 20], type=int, help='window size')
    parser.add_argument("--anomaly_ratio", nargs="+", default=[1.0], type=float)  # [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.13, 0.15]
    parser.add_argument('--clip', dest='clip', default=3.0, type=float, help='Gradient clipping')
    parser.add_argument('--num_epochs', dest='num_epochs', default=100, type=int, help='total epoch number')
    parser.add_argument('--batch_size', dest='batch_size', default=128, type=int, help='Batch size')
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument('--ffn_dim', dest='ffn_dim', default=768, type=int, help='Feed forward network dimension')
    parser.add_argument('--output_dim', dest='output_dim', default=1, type=int, help='Output dimension')
    parser.add_argument("--max_hop", type=int, default=10)
    parser.add_argument('--num_layer', dest='num_layer', default=3, type=int, help='Encoder layer number')
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--perturb_noise", default=0.00, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--peak_lr", default=2e-4, type=float)
    parser.add_argument("--end_lr", default=1e-9, type=float)
    parser.add_argument("--warmup_epoch", default=3, type=int)
    parser.add_argument('--dropout', dest='dropout', default=0.3, type=float, help='Dropout rate')
    parser.add_argument('--patience', dest='patience', default=20, type=int, help='Early stopping after patience')
    parser.add_argument('--seed', dest='seed', type=int, default=42, help='seed')
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

    best_threshold = 0.0  # get from validation set and will be used for testing data

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=args.patience, delta=1e-6, verbose=False, path=f'{args.model}_checkpoint.pt')

    for epoch in range(args.num_epochs):

        ######################
        # training the model #
        ######################
        pred_label = np.array([])
        y_label = np.array([])

        train_pred_loss = []
        model.train()
        criterion = F.binary_cross_entropy_with_logits

        for batch_idx, batch_data in enumerate(train_loader):
            train_begin_time = time.time()
            batch_data = batch_data.to(device)
            batch_data.requires_grad_(False)

            output = model(batch_data)
            labels = batch_data.y.view(output.shape).to(device)
            loss = criterion(output, labels.float())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            elapsed = time.time() - train_begin_time
            train_time_list.append(elapsed)
            train_pred_loss.append(loss.cpu().detach().numpy().flatten())

            pred_label = np.concatenate((pred_label, output.cpu().detach().numpy().flatten()))
            y_label = np.concatenate((y_label, batch_data.y.cpu().detach().numpy().flatten()))

        train_loss = np.array(train_pred_loss)

        precision, recall, threshold = precision_recall_curve(y_label, pred_label)
        F1_score = np.divide(2 * precision * recall,
                             precision + recall,
                             out=np.zeros(precision.shape, dtype=float),
                             where=(precision + recall) != 0)

        best_threshold = threshold[np.argmax(F1_score)]

        ######################
        # validate the model #
        ######################
        model.eval()
        valid_pred = []
        for batch_idx, batch_data in enumerate(valid_loader):
            batch_data = batch_data.to(device)
            batch_data.requires_grad_(False)

            output = model(batch_data)
            labels = batch_data.y.view(output.shape).to(device)
            loss = criterion(output, labels.float())
            valid_pred.append(loss.cpu().detach().numpy().flatten())

            # pred_label = np.concatenate((pred_label, output.cpu().detach().numpy().flatten()))
            # y_label = np.concatenate((y_label, batch_data.y.cpu().detach().numpy().flatten()))

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        valid_loss = np.array(valid_pred)

        # precision, recall, threshold = precision_recall_curve(y_label, pred_label)
        # F1_score = np.divide(2 * precision * recall,
        #                      precision + recall,
        #                      out=np.zeros(precision.shape, dtype=float),
        #                      where=(precision + recall) != 0)
        # best_threshold = threshold[np.argmax(F1_score)]

        print(f'[{epoch+1}/{args.num_epochs}] train_loss: {train_loss.mean():.7f} valid_loss: {valid_loss.mean():.7f}')

        early_stopping(valid_loss.mean(), model, best_threshold)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    ######################
    # test the model #
    ######################

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(f'{args.model}_checkpoint.pt'))

    model.eval()
    test_pred = []
    y = []

    for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="testing")):
        test_begin_time = time.time()
        batch_data = batch_data.to(device)
        # batch_data.requires_grad_(False)

        output = model(batch_data)
        test_pred.append(output.cpu().detach().numpy().flatten())

        y.append(batch_data.y.item())
        elapsed = time.time() - test_begin_time
        test_time_list.append(elapsed)

    ground_truth = np.array(y)
    pred_label = np.array(test_pred)

    fpr_ab, tpr_ab, _ = roc_curve(ground_truth, pred_label)
    test_roc_ab = auc(fpr_ab, tpr_ab)

    precision, recall, F1_score, _ = precision_recall_fscore_support(ground_truth, (pred_label >= early_stopping.get_threshold()).astype(int), average='binary')
    print(f'auc-roc={test_roc_ab}, precision={precision}, recall={recall}, f1-score={F1_score}, thresh_hold={best_threshold}\n')

    with open(f'{args.model}_Experiment_results.txt', 'a+') as f:
        print(f"precision={precision:.5f}, recall={recall:.5f}, f1-score={F1_score:.5f}, auc-roc={test_roc_ab}\n", file=f)

    #     print(f"Total training time({args.num_epochs} epochs) = {np.array(train_time_list).sum():.3f}(s)", file=f)
    #     print(f"Average training time per epoch = {(np.array(train_time_list).sum()/args.num_epochs):.3f)}(s)", file=f)
    #     print(f"Average testing time per graph = {np.array(test_time_list).mean() * 1000:.3f}(ms)\n", file=f)
    #
    # print(f"Total training time({args.num_epochs} epochs) = {np.array(train_time_list).sum():.3f}(s)")
    # print(f"Average training time per epoch = {(np.array(train_time_list).sum()/args.num_epochs):.3f}(s)")
    # print(f"Average testing time per graph = {np.array(test_time_list).mean() * 1000:.3f}(ms)\n")

    return test_roc_ab, precision, recall, F1_score


def get_data_directory(data_name, data_dir, ratio_set):
    return f"{data_dir}/{data_name}/{data_name.lower()}_{ratio_set}_tar/"


def set_param_configuration(data_name, data_directory, feature_type='semantics', w_size='session', s_size=None,
                            embedding='bert'):
    para_config = {
        "data_dir": data_directory,
        "dataset": data_name,
        "feature_type": feature_type,  # "semantics", "non_semantics"
        "window_type": "session" if data_name == 'hdfs' else "sliding",
        "use_tfidf": False if embedding == 'bert' else True,
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

    for data_set in args.datasets:
        for test_ratio in [0.2]:
            if data_set == 'hdfs':
                windows = ['session']
            else:
                windows = args.window_size
            for window_size in windows:

                data_dir = get_data_directory(data_set, args.data_dir, test_ratio)
                params = set_param_configuration(data_set, data_dir, args.feature_type, window_size, args.embedding_type)
                print(f"Starting to process dataset={params['dataset']} window_size={params['window_size']}")
                print("*" * 100)

                tr_graphs = LogGraphDataset(data_name=params['dataset'], supervise_mode='mixed',
                                            window_size=params['window_size'], feature_type=params['feature_type'],
                                            use_tfidf=params['use_tfidf'], raw_dir=data_dir, save_dir=data_dir,
                                            dataset_type='train', verbose=False)

                ts_graphs = LogGraphDataset(data_name=params['dataset'], supervise_mode='mixed',
                                            window_size=params['window_size'], feature_type=params['feature_type'],
                                            use_tfidf=params['use_tfidf'], raw_dir=data_dir, save_dir=data_dir,
                                            dataset_type='test', verbose=False)

                node_attr_dim = tr_graphs.feat_dim
                graphs_test = ts_graphs
                # use generator to save memories
                te_anomaly_num = sum(1 for (graph, label) in graphs_test if label == 1)

                for sample_num in args.sampling_training:
                    for anomaly_ratio in args.anomaly_ratio:
                        sample_num = float(sample_num)
                        training_data, training_labels = tr_graphs.get_samples(sample_size=sample_num, anomaly_ratio=anomaly_ratio)

                        with open(f'{args.model}_Experiment_results.txt', 'a+') as f:
                            print(
                                f"Model={args.model}, Dataset={params['dataset']}, Window_size={params['window_size']}, Embedding_type={args.embedding_type}, "
                                f"Epochs={args.num_epochs}, Patience={args.patience}, Max_hops={args.max_hop}\n"
                                f"Test_ratio={test_ratio}, sampling_ratio={sample_num}, anomaly_ratio={anomaly_ratio}, "
                                f"Dataset_nodes={tr_graphs.dataset_node_size}, Max_graph_nodes={tr_graphs.max_graph_node_size}"
                                , file=f)
                            print('*' * 100, file=f)

                        print(
                            f"Model={args.model}, Dataset={params['dataset']}, Window_size={params['window_size']}, Embedding_type={args.embedding_type}, "
                            f"Epochs={args.num_epochs}, Patience={args.patience}, Max_hops={args.max_hop}\n"
                            f"Test_ratio={test_ratio}, sampling_ratio={sample_num}, anomaly_ratio={anomaly_ratio}, "
                            f"Dataset_nodes={tr_graphs.dataset_node_size}, Max_graph_nodes={tr_graphs.max_graph_node_size}")
                        print('*' * 100)

                        result_auc_list = []
                        result_precision_list = []
                        result_recall_list = []
                        result_f1_list = []

                        # kfd = StratifiedKFold(n_splits=3, random_state=args.seed, shuffle=True)
                        kfd = StratifiedShuffleSplit(n_splits=3, train_size=int(len(training_labels)*0.8), random_state=args.seed)

                        for tr_index, va_index in kfd.split(training_data, training_labels):

                            graphs_train = [training_data[idx] for idx in tr_index]
                            graphs_valid = [training_data[idx] for idx in va_index]

                            # use generator to save memories
                            tr_anomaly_num = sum(1 for (graph, label) in graphs_train if label == 1)
                            va_anomaly_num = sum(1 for (graph, label) in graphs_valid if label == 1)

                            with open(f'{args.model}_Experiment_results.txt', 'a+') as f:
                                print(f"Training set size={len(graphs_train)}, "
                                      f"Validation set size={len(graphs_valid)}, "
                                      f"Testing set size={len(graphs_test)}\n"
                                      f"training set anomalies={tr_anomaly_num}({(tr_anomaly_num/len(graphs_train)):.3f}), "
                                      f"Validation set anomalies=={va_anomaly_num}({(va_anomaly_num/len(graphs_valid)):.3f}), "
                                      f"testing set anomalies={te_anomaly_num}({(te_anomaly_num/len(graphs_test)):.3f})\n"
                                      , file=f)

                            print(f"Training set size={len(graphs_train)}, "
                                  f"Validation set size={len(graphs_valid)}, "
                                  f"Testing set size={len(graphs_test)}\n"
                                  f"training set anomalies={tr_anomaly_num}({(tr_anomaly_num/len(graphs_train)):.3f}), "
                                  f"Validation set anomalies=={va_anomaly_num}({(va_anomaly_num/len(graphs_valid)):.3f}), "
                                  f"testing set anomalies={te_anomaly_num}({(te_anomaly_num/len(graphs_test)):.3f})\n")

                            dataset_feature_train = GraphFeatureExtractor(graphs_train,
                                                                          max_dataset_nodes=tr_graphs.dataset_node_size,
                                                                          max_graph_nodes=tr_graphs.max_graph_node_size,
                                                                          node_attr_dim=node_attr_dim,
                                                                          transform=Compose(
                                                                              [ShortestPathGenerator(), OneHotEdgeAttr()]),
                                                                          )

                            dataset_feature_valid = GraphFeatureExtractor(graphs_valid,
                                                                          max_dataset_nodes=tr_graphs.dataset_node_size,
                                                                          max_graph_nodes=tr_graphs.max_graph_node_size,
                                                                          node_attr_dim=node_attr_dim,
                                                                          transform=Compose(
                                                                              [ShortestPathGenerator(), OneHotEdgeAttr()]),
                                                                          )

                            dataset_feature_test = GraphFeatureExtractor(graphs_test,
                                                                         max_dataset_nodes=ts_graphs.dataset_node_size,
                                                                         max_graph_nodes=ts_graphs.max_graph_node_size,
                                                                         node_attr_dim=node_attr_dim,
                                                                         transform=Compose(
                                                                             [ShortestPathGenerator(), OneHotEdgeAttr()]),
                                                                         )

                            model = GRPENetwork(out_dim=args.output_dim, d_model=args.embedding_dim,
                                                dim_feedforward=args.ffn_dim,
                                                num_layer=args.num_layer, nhead=args.nhead, max_hop=args.max_hop,
                                                num_node_type=-node_attr_dim,
                                                num_edge_type=128,
                                                perturb_noise=args.perturb_noise, dropout=args.dropout,
                                                ).to(device)

                            collate_fn = LogCollator()

                            data_train_loader = DataLoader(dataset_feature_train,
                                                           shuffle=True,
                                                           batch_size=args.batch_size,
                                                           collate_fn=collate_fn)

                            data_valid_loader = DataLoader(dataset_feature_valid,
                                                           shuffle=True,
                                                           batch_size=args.batch_size,
                                                           collate_fn=collate_fn)

                            data_test_loader = DataLoader(dataset_feature_test,
                                                          shuffle=False,
                                                          batch_size=1,
                                                          collate_fn=collate_fn)

                            result = train(data_train_loader, data_valid_loader, data_test_loader, model, args)

                            result_auc_list.append(result[0])
                            result_precision_list.append(result[1])
                            result_recall_list.append(result[2])
                            result_f1_list.append(result[3])

                            del data_train_loader, data_valid_loader, data_test_loader, dataset_feature_train, dataset_feature_valid, dataset_feature_test
                            del graphs_train, graphs_valid

                        del training_data, training_labels

                        record_result(args.model, data_set, window_size, result_auc_list, result_precision_list, result_recall_list, result_f1_list)
