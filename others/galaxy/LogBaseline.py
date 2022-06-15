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

from sklearn.metrics import auc, precision_recall_curve, roc_curve, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from LogGraphDataset import LogGraphDataset
from feature_extraction import GraphFeatureExtractor, device
from tansform import ShortestPathGenerator, OneHotEdgeAttr, LogCollator
from evaluate import record_result
from model.graph_gat2gcn import GAT2Network
from model.graph_transformergcn import TransformerGCNNetwork
from model.graph_gcn import GENGCNNetwork
from model.graph_gatedgcn import GatedGCNNetwork

from model.lr import PolynomialDecayLR
from early_stopping import EarlyStopping

np.set_printoptions(threshold=np.inf)

def arg_parse():
    parser = argparse.ArgumentParser(description='LogGT Arguments')
    parser.add_argument('--data_dir', dest='data_dir', default='../dataset/processed/',
                        help='Directory where data is located')
    parser.add_argument('--model', default="transformer_cov", choices=['gated_cov', 'graph_cov', 'gat2_cov', 'transformer_cov'], help='The model to be used')
    parser.add_argument('--datasets', nargs="+", default=['hdfs', 'bgl', 'spirit', 'tbd'], help='The dataset to be processed')
    parser.add_argument('--normal_only', dest='normal_only', action='store_true', default=False,
                        help='Whether normal only data are used for training')
    parser.add_argument("--sampling_training", default=False, action="store_true", help='whether to train the model with sampling data')
    parser.add_argument('--feature_type', dest='feature_type', default='semantics', help='use what node feature')
    parser.add_argument('--use_tfidf', dest='use_tfidf', default=False, help='Whether tfidf is used')
    # parser.add_argument("--anomaly_ratio", choices=[0.0, 0.03, 0.1, 0.3, 0.5, 1.0], default=0.03, type=float)
    parser.add_argument('--clip', dest='clip', default=3.0, type=float, help='Gradient clipping.')
    parser.add_argument('--num_epochs', dest='num_epochs', default=50, type=int, help='total epoch number')
    parser.add_argument('--batch_size', dest='batch_size', default=128, type=int, help='Batch size')
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument('--ffn_dim', dest='ffn_dim', default=768, type=int, help='Feed forward network dimension')
    parser.add_argument('--output_dim', dest='output_dim', default=1, type=int, help='Output dimension')
    parser.add_argument("--max_hop", type=int, default=10)
    parser.add_argument('--num_layer', dest='num_layer', default=3, type=int, help='Encoder layer number')
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--use_independent_token", default=False, action="store_true")
    parser.add_argument("--perturb_noise", default=0.00, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--peak_lr", default=2e-4, type=float)
    parser.add_argument("--end_lr", default=1e-9, type=float)
    parser.add_argument("--warmup_epoch", default=3, type=int)
    parser.add_argument('--dropout', dest='dropout', default=0.3, type=float, help='Dropout rate')
    parser.add_argument('--patience', dest='patience', default=10, type=int, help='Early stopping after patience')
    parser.add_argument('--seed', dest='seed', type=int, default=1, help='seed')
    return parser.parse_args()

def train(train_loader, valid_loader, test_loader, student_model, args):
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, student_model.parameters()), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, student_model.parameters()),
                                  lr=args.peak_lr, weight_decay=args.weight_decay)
    scheduler = PolynomialDecayLR(optimizer, warmup_updates=args.warmup_epoch * len(train_loader),
                                  tot_updates=args.num_epochs * len(train_loader), lr=args.peak_lr,
                                  end_lr=args.end_lr, power=1.0)

    train_time_list = []
    test_time_list = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=args.patience, delta=1e-4, verbose=False)

    for epoch in range(args.num_epochs):

        ######################
        # training the model #
        ######################

        train_pred_loss = []
        student_model.train()

        for batch_idx, batch_data in enumerate(train_loader):
            train_begin_time = time.time()
            batch_data = batch_data.to(device)
            batch_data.requires_grad_(False)

            output = student_model(batch_data)
            labels = batch_data.y.view(output.shape).to(device)
            loss = F.binary_cross_entropy_with_logits(output, labels.float())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student_model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            elapsed = time.time() - train_begin_time
            train_time_list.append(elapsed)
            train_pred_loss.append(loss.cpu().detach().numpy().flatten())

        train_loss = np.array(train_pred_loss)

        ######################
        # validate the model #
        ######################
        student_model.eval()
        valid_pred = []
        for batch_idx, batch_data in enumerate(valid_loader):
            batch_data = batch_data.to(device)
            batch_data.requires_grad_(False)

            output = student_model(batch_data)
            labels = batch_data.y.view(output.shape).to(device)
            loss = F.binary_cross_entropy_with_logits(output, labels.float())
            valid_pred.append(loss.cpu().detach().numpy().flatten())

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        valid_loss = np.array(valid_pred)

        print(f'[{epoch+1}/{args.num_epochs}] train_loss: {train_loss.mean():.5f} valid_loss: {valid_loss.mean():.5f}')

        early_stopping(valid_loss.mean(), student_model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    ######################
    # test the model #
    ######################

    # load the last checkpoint with the best model
    student_model.load_state_dict(torch.load('checkpoint.pt'))

    student_model.eval()
    test_pred = []
    y = []

    for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="testing")):
        test_begin_time = time.time()
        batch_data = batch_data.to(device)
        batch_data.requires_grad_(False)

        output = student_model(batch_data)
        test_pred.append(output.cpu().detach().numpy().flatten())

        y.append(batch_data.y.item())
        elapsed = time.time() - test_begin_time
        test_time_list.append(elapsed)

    ground_truth = np.array(y)
    pred_label = np.array(test_pred)

    if args.normal_only:
        print(f'\nAnomaly detection using normal data only training:')
    else:
        print(f'\nAnomaly detection using mixed data training:')

    fpr_ab, tpr_ab, _ = roc_curve(ground_truth, pred_label)
    test_roc_ab = auc(fpr_ab, tpr_ab)

    precision, recall, threshold = precision_recall_curve(ground_truth, pred_label)
    F1_score = np.divide(2 * precision * recall,
                         precision + recall,
                         out=np.zeros(precision.shape, dtype=float),
                         where=(precision + recall) != 0)

    # precision = precision_score(ground_truth, pred_label)
    # recall = recall_score(ground_truth, pred_label)
    # F1_score = f1_score(ground_truth, pred_label)
    print(f'auc-roc={test_roc_ab:.3f}, '
          f'precision={precision[np.argmax(F1_score)]}, '
          f'recall={recall[np.argmax(F1_score)]}, '
          f'f1-score={np.max(F1_score)}\n')

    aucroc_final = test_roc_ab
    precision_final = precision[np.argmax(F1_score)]
    recall_final = recall[np.argmax(F1_score)]
    f1_score_final = np.max(F1_score)

    with open(f'{args.model}_Experiment_results.txt', 'a+') as f:
        print(f"precision={precision_final:.3f}, recall={recall_final:.3f}, f1-score={np.max(F1_score):.3f}"
              f", threshold={f1_score_final:.3f}\n", file=f)
    #     print(f"Total training time({args.num_epochs} epochs) = {np.array(train_time_list).sum():.3f}(s)", file=f)
    #     print(f"Average training time per epoch = {(np.array(train_time_list).sum()/args.num_epochs):.3f)}(s)", file=f)
    #     print(f"Average testing time per graph = {np.array(test_time_list).mean() * 1000:.3f}(ms)\n", file=f)
    #
    # print(f"Total training time({args.num_epochs} epochs) = {np.array(train_time_list).sum():.3f}(s)")
    # print(f"Average training time per epoch = {(np.array(train_time_list).sum()/args.num_epochs):.3f}(s)")
    # print(f"Average testing time per graph = {np.array(test_time_list).mean() * 1000:.3f}(ms)\n")

    return aucroc_final, precision_final, recall_final, f1_score_final


def get_data_directory(data_name, data_dir, ratio_set):
    return f"{data_dir}/{data_name}/{data_name.lower()}_{ratio_set}_tar/"


def set_param_configuration(data_name, data_directory, feature_type='semantics', w_size='session', s_size=None,
                            w_tfidf=False):
    para_config = {
        "data_dir": data_directory,
        "dataset": data_name,
        "feature_type": feature_type,  # "semantics", "non_semantics"
        "window_type": "session" if data_name == 'hdfs' else "sliding",
        "use_tfidf": w_tfidf,
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
                windows = [100, 20]
            for window_size in windows:

                data_dir = get_data_directory(data_set, args.data_dir, test_ratio)
                params = set_param_configuration(data_set, data_dir, args.feature_type, window_size, args.use_tfidf)
                print(f"Starting to process dataset={params['dataset']} window_size={params['window_size']}")
                print("*" * 100)

                mode = 'normal' if args.normal_only else 'mixed'

                tr_graphs = LogGraphDataset(data_name=params['dataset'], supervise_mode=mode,
                                            window_size=params['window_size'], feature_type=params['feature_type'],
                                            use_tfidf=params['use_tfidf'], raw_dir=data_dir, save_dir=data_dir,
                                            dataset_type='train', verbose=False)

                ts_graphs = LogGraphDataset(data_name=params['dataset'], supervise_mode=mode,
                                            window_size=params['window_size'], feature_type=params['feature_type'],
                                            use_tfidf=params['use_tfidf'], raw_dir=data_dir, save_dir=data_dir,
                                            dataset_type='test', verbose=False)

                max_graph_nodes_num = tr_graphs.max_graph_node_size
                max_dataset_node_num = tr_graphs.dataset_node_size
                node_attr_dim = tr_graphs.feat_dim

                if args.sampling_training:
                    training_data_sampling_ratio = [0.8, 0.5, 0.3, 0.1]
                else:
                    training_data_sampling_ratio = [1.0]

                for sample_num in training_data_sampling_ratio:
                    for anomaly_ratio in [0.1]:
                        if args.sampling_training:
                            training_data, training_labels = tr_graphs.get_samples(sample_size=sample_num, anomaly_ratio=anomaly_ratio)
                        else:
                            training_data, training_labels = tr_graphs, tr_graphs.labels

                        with open('Experiment_results.txt', 'a+') as f:
                            print(
                                f"Model={args.model}, Dataset={params['dataset']}, Window_size={params['window_size']}, Training_mode={mode}, "
                                f"Epochs={args.num_epochs}, Patience={args.patience}, Feature type={params['feature_type']}\n"
                                f"Test_ratio={test_ratio}, sampling_ratio={sample_num}, anomaly_ratio={anomaly_ratio}, "
                                f"Dataset_nodes={tr_graphs.dataset_node_size}, Max_graph_nodes={tr_graphs.max_graph_node_size}"
                                , file=f)
                            print('*' * 100, file=f)

                        print(
                            f"Model={args.model}, Dataset={params['dataset']}, Window_size={params['window_size']}, Training_mode={mode}, "
                            f"Epochs={args.num_epochs}, Patience={args.patience}, Feature type={params['feature_type']}\n"
                            f"Test_ratio={test_ratio}, sampling_ratio={sample_num}, anomaly_ratio={anomaly_ratio}, "
                            f"Dataset_nodes={tr_graphs.dataset_node_size}, Max_graph_nodes={tr_graphs.max_graph_node_size}")
                        print('*' * 100)

                        result_auc_list = []
                        result_precision_list = []
                        result_recall_list = []
                        result_f1_list = []

                        kfd = StratifiedKFold(n_splits=3, random_state=args.seed, shuffle=True)

                        for tr_index, va_index in kfd.split(training_data, training_labels):

                            graphs_train = [training_data[idx] for idx in tr_index]
                            graphs_valid = [training_data[idx] for idx in va_index]
                            graphs_test = ts_graphs

                            tr_anomaly_num = sum([1 for (graph, label) in graphs_train if label == 1])
                            va_anomaly_num = sum([1 for (graph, label) in graphs_valid if label == 1])
                            te_anomaly_num = sum([1 for (graph, label) in graphs_test if label == 1])

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
                                                                          max_dataset_nodes=max_dataset_node_num,
                                                                          max_graph_nodes=max_graph_nodes_num,
                                                                          node_attr_dim=node_attr_dim,
                                                                          transform=Compose(
                                                                              [ShortestPathGenerator(), OneHotEdgeAttr()]),
                                                                          )

                            dataset_feature_valid = GraphFeatureExtractor(graphs_valid,
                                                                          max_dataset_nodes=max_dataset_node_num,
                                                                          max_graph_nodes=max_graph_nodes_num,
                                                                          node_attr_dim=node_attr_dim,
                                                                          transform=Compose(
                                                                              [ShortestPathGenerator(), OneHotEdgeAttr()]),
                                                                          )

                            dataset_feature_test = GraphFeatureExtractor(graphs_test,
                                                                         max_dataset_nodes=max_dataset_node_num,
                                                                         max_graph_nodes=max_graph_nodes_num,
                                                                         node_attr_dim=node_attr_dim,
                                                                         transform=Compose(
                                                                             [ShortestPathGenerator(), OneHotEdgeAttr()]),
                                                                         )

                            if args.model == 'gated_cov':

                                model = GatedGCNNetwork(out_dim=args.output_dim, d_model=args.embedding_dim,
                                                        num_layer=args.num_layer, max_hop=args.max_hop,
                                                        num_node_type=-node_attr_dim,
                                                        perturb_noise=args.perturb_noise,
                                                        ).to(device)
                            elif args.model == 'graph_cov':
                                model = GENGCNNetwork(out_dim=args.output_dim, d_model=args.embedding_dim,
                                                   num_layer=args.num_layer,
                                                   num_node_type=-node_attr_dim,
                                                   perturb_noise=args.perturb_noise,
                                                   ).to(device)

                            elif args.model == 'gat2_cov':
                                model = GAT2Network(out_dim=args.output_dim, d_model=args.embedding_dim,
                                                    num_layer=args.num_layer, nhead=args.nhead,
                                                    num_node_type=-node_attr_dim,
                                                    perturb_noise=args.perturb_noise, dropout=args.dropout,
                                                    ).to(device)

                            elif args.model == 'transformer_cov':
                                model = TransformerGCNNetwork(out_dim=args.output_dim, d_model=args.embedding_dim,
                                                              num_layer=args.num_layer, nhead=args.nhead,
                                                              num_node_type=-node_attr_dim,
                                                              perturb_noise=args.perturb_noise, dropout=args.dropout,
                                                              ).to(device)
                            else:
                                raise ValueError(f"The specified model {args.model} is not supported")

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

                        record_result(args.model, data_set, result_auc_list, result_precision_list, result_recall_list, result_f1_list)
