# -*- coding: utf-8 -*-
"""
Created on Tue April 07 17:16:36 2022

@author: Yongzheng Xie
"""

import time
import argparse
import random
import numpy as np
from numpy.random import seed

import scipy as stats

import torch
import torch.nn as nn
import torch.backends.cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.transforms import Compose

from sklearn.metrics import auc, precision_recall_curve, roc_curve, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

from LogGraphDataset import LogGraphDataset
from feature_extraction import GraphFeatureExtractor, graph_indicator, label_indicator, device
from tansform import ShortestPathGenerator, OneHotEdgeAttr, LogCollator
from evaluate import record_result
from grpe.model.graph_self_attention import GRPENetwork
from grpe.lr import PolynomialDecayLR

np.set_printoptions(threshold=np.inf)

def arg_parse():
    parser = argparse.ArgumentParser(description='LogGT Arguments')
    parser.add_argument('--data_dir', dest='data_dir', default='./dataset/processed/',
                        help='Directory where benchmark is located')
    parser.add_argument('--normal_only', dest='normal_only', action='store_true', default=False,
                        help='Whether normal only data are used for training')
    parser.add_argument('--feature_type', dest='feature_type', default='semantics', help='use what node feature')
    parser.add_argument('--use_tfidf', dest='use_tfidf', default=False, help='Whether tfidf is used')
    parser.add_argument("--anomaly_ratio", choices=[0.0, 0.03, 0.1, 0.3, 0.5, 1.0], default=1.0, type=float)
    parser.add_argument('--clip', dest='clip', default=2.0, type=float, help='Gradient clipping.')
    parser.add_argument('--loss_use', dest='loss_use', choices=["node_loss", "graph_loss", "node_graph_loss"],
                        default='node_graph_loss', type=str, help='which loss is used.')
    parser.add_argument('--num_epochs', dest='num_epochs', default=30, type=int, help='total epoch number')
    parser.add_argument('--batch_size', dest='batch_size', default=64, type=int, help='Batch size')
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument('--ffn_dim', dest='ffn_dim', default=768, type=int, help='Feed forward network dimension')
    parser.add_argument('--output_dim', dest='output_dim', default=1, type=int, help='Output dimension')
    parser.add_argument("--max_hop", type=int, default=10)
    parser.add_argument('--num_layer', dest='num_layer', default=3, type=int, help='Encoder layer number')
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--use_independent_token", default=False, action="store_true")
    parser.add_argument("--perturb_noise", default=0.0, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--peak_lr", default=2e-4, type=float)
    parser.add_argument("--end_lr", default=1e-9, type=float)
    parser.add_argument("--warmup_epoch", default=3, type=int)
    parser.add_argument('--dropout', dest='dropout', default=0.3, type=float, help='Dropout rate')
    parser.add_argument('--seed', dest='seed', type=int, default=1, help='seed')
    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def init_center_c(input_dim, train_loader, model, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = torch.zeros(input_dim, device=device)

    model.eval()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            inputs = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    print(f"Initialize c successfully")

    return c

def get_radius(dist, nu):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


# calculateMahalanobis function to calculate
# the Mahalanobis distance
def calculateMahalanobis(y=None, data=None, cov=None):
    y_mu = y - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)
    return mahal.diagonal()

def train(train_loader, test_loader, teacher_model, student_model, args):
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, student_model.parameters()), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, student_model.parameters()),
                                  lr=args.peak_lr, weight_decay=args.weight_decay)
    scheduler = PolynomialDecayLR(optimizer, warmup_updates=args.warmup_epoch * len(train_loader),
                                  tot_updates=args.num_epochs * len(train_loader), lr=args.peak_lr,
                                  end_lr=args.end_lr, power=1.0)

    aucroc_final = 0
    precision_final = 0
    recall_final = 0
    f1_score_final = 0
    train_time_list = []
    test_time_list = []

    c_param = init_center_c(args.output_dim, train_loader, student_model)
    nu_param = 0.01
    r_param = 0.0

    for epoch in range(args.num_epochs):
        train_pred_loss = []
        student_model.train()

        for batch_idx, batch_data in enumerate(train_loader):
            train_begin_time = time.time()
            batch_data = batch_data.to(device)
            batch_data.requires_grad_(False)

            # embed = student_model(batch_data)
            # labels = batch_data.y.view(embed.shape).to(device)
            # loss = F.binary_cross_entropy_with_logits(embed, labels.float())

            output = student_model(batch_data)
            loss = torch.mean((torch.sigmoid(output.squeeze()) - c_param) ** 2)
            # loss = torch.mean(torch.sum((output - c_param) ** 2, dim=1))

            # scores = output - r_param ** 2
            # loss = r_param ** 2 + (1 / nu_param) * torch.mean(torch.max(torch.zeros_like(scores), scores))
            # r_param = torch.tensor(get_radius(output, nu_param), device=device)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student_model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            elapsed = time.time() - train_begin_time
            train_time_list.append(elapsed)
            train_pred_loss.append(loss.cpu().detach().numpy())

        train_loss = np.array(train_pred_loss)
        print(f"Training: Epoch={epoch+1}/{args.num_epochs}, training loss={train_loss.mean()}")

        if (epoch + 1) % 10 == 0 and epoch > 0:
            student_model.eval()
            test_pred = []
            y = []

            for batch_idx, batch_data in enumerate(test_loader):
                test_begin_time = time.time()
                batch_data = batch_data.to(device)
                batch_data.requires_grad_(False)

                # embed = student_model(batch_data)
                # test_pred.append(embed.cpu().detach().numpy().flatten())

                embed = student_model(batch_data)
                scores = (torch.sigmoid(embed.squeeze()) - c_param) ** 2
                # scores = torch.mean(torch.sum((embed - c_param) ** 2, dim=1))

                # scores = output - r_param ** 2
                # scores = r_param ** 2 + (1 / nu_param) * torch.mean(torch.max(torch.zeros_like(scores), scores))

                test_pred.append(scores.cpu().detach().numpy().flatten())

                y.append(batch_data.y.item())
                elapsed = time.time() - test_begin_time
                test_time_list.append(elapsed)

            # print(f"testing: max={np.max(np.array(em_list))}, min={np.min(np.array(em_list))}, mean={np.mean(np.array(em_list))}")
            ground_truth = np.array(y)
            pred_label = np.array(test_pred)

            if args.normal_only:
                print(f'\nsemi-supervised anomaly detection:')
            else:
                print(f'\nun-supervised anomaly detection:')

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

            with open('Experiment_results.txt', 'a+') as f:
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


if __name__ == '__main__':

    kfold = 3
    args = arg_parse()
    setup_seed(args.seed)

    for data_set in ['hdfs', 'bgl', 'spirit']:
        for test_ratio in [0.2]:
            if data_set == 'hdfs':
                windows = ['session']
            else:
                windows = [100, 20]
            for window_size in windows:
                result_auc_list = []
                result_precision_list = []
                result_recall_list = []
                result_f1_list = []

                data_dir = get_data_directory(data_set, args.data_dir, test_ratio)
                params = set_param_configuration(data_set, data_dir, args.feature_type, window_size, args.use_tfidf)
                print(f"Starting to process dataset={params['dataset']} window_size={params['window_size']}")
                print("*" * 100)

                mode = 'normal' if args.normal_only else 'mixed'

                graphs = LogGraphDataset(data_name=params['dataset'], supervise_mode=mode,
                                         window_size=params['window_size'], feature_type=params['feature_type'],
                                         use_tfidf=params['use_tfidf'], raw_dir=data_dir, save_dir=data_dir,
                                         dataset_type='test', anomaly_ratio=args.anomaly_ratio, verbose=False)

                max_graph_nodes_num = graphs.max_graph_node_size
                max_dataset_node_num = graphs.dataset_node_size
                node_attr_dim = graphs.feat_dim
                with open('Experiment_results.txt', 'a+') as f:
                    print(f"Dataset={params['dataset']}, Window_size={params['window_size']}, Training_mode={mode}, "
                          f"Epochs={args.num_epochs}, Loss_use={args.loss_use}, Anomaly_ratio={args.anomaly_ratio}\n"
                          f"Feature type={params['feature_type']}, Use_tfidf= {params['use_tfidf']}, Test_ratio={test_ratio}, "
                          f"Dataset_node_num={graphs.dataset_node_size}, Max_graph_node_num={graphs.max_graph_node_size}"
                          , file=f)
                    print('*' * 100, file=f)

                print(
                    f"Dataset={params['dataset']}, Window_size={params['window_size']}, Mode={mode}, "
                    f"Epochs={args.num_epochs}, Loss_use={args.loss_use}, Anomaly_ratio={args.anomaly_ratio}\n"
                    f"Feature_type={params['feature_type']}, Use_tfidf= {params['use_tfidf']}, Test_ratio={test_ratio}, "
                    f"Dataset_node_num={graphs.dataset_node_size}, Max_graph_node_num={graphs.max_graph_node_size}")
                print('*' * 100)

                kfd = StratifiedKFold(n_splits=kfold, random_state=args.seed, shuffle=True)

                for k, (train_index, test_index) in enumerate(kfd.split(graphs, graphs.labels)):

                    graphs_train_all = [graphs[i] for i in train_index]
                    graphs_test = [graphs[i] for i in test_index]

                    if args.normal_only:
                        # training data just includes normal graphs
                        graphs_train = [graph for graph in graphs_train_all if graph[label_indicator] == 0]
                        tr_anomaly_num = 0
                    else:
                        # training data includes anomalies
                        graphs_train = graphs_train_all
                        tr_anomaly_num = sum([1 for graph in graphs_train_all if graph[label_indicator] == 1])

                    te_anomaly_num = sum([1 for graph in graphs_test if graph[label_indicator] == 1])

                    with open('Experiment_results.txt', 'a+') as f:
                        print(f"Total data size={len(graphs)}, "
                              f"Training size={len(graphs_train)}, "
                              f"Testing size={len(graphs_test)}, "
                              f"training set anomalies={tr_anomaly_num}, "
                              f"testing set anomalies={te_anomaly_num}\n"
                              , file=f)

                    print(f"Total data size={len(graphs)}, "
                          f"Training size={len(graphs_train)}, "
                          f"Testing size={len(graphs_test)}, "
                          f"Training set anomalies={tr_anomaly_num}, "
                          f"Testing set anomalies={te_anomaly_num}\n")

                    dataset_feature_train = GraphFeatureExtractor(graphs_train,
                                                                  max_dataset_nodes=max_dataset_node_num,
                                                                  max_graph_nodes=max_graph_nodes_num,
                                                                  node_attr_dim=node_attr_dim,
                                                                  transform=Compose(
                                                                      [ShortestPathGenerator(), OneHotEdgeAttr()]),
                                                                  )

                    model_teacher = GRPENetwork(out_dim=args.output_dim, d_model=args.embedding_dim,
                                                dim_feedforward=args.ffn_dim,
                                                num_layer=args.num_layer, nhead=args.nhead, max_hop=args.max_hop,
                                                num_node_type=-node_attr_dim,
                                                num_edge_type=1,
                                                use_independent_token=args.use_independent_token,
                                                perturb_noise=args.perturb_noise, dropout=args.dropout,
                                                ).to(device)
                    for param in model_teacher.parameters():
                        param.requires_grad = False

                    model_student = GRPENetwork(out_dim=args.output_dim, d_model=args.embedding_dim,
                                                dim_feedforward=args.ffn_dim,
                                                num_layer=args.num_layer, nhead=args.nhead, max_hop=args.max_hop,
                                                num_node_type=-node_attr_dim,
                                                num_edge_type=1,
                                                use_independent_token=args.use_independent_token,
                                                perturb_noise=args.perturb_noise, dropout=args.dropout,
                                                ).to(device)

                    collate_fn = LogCollator()

                    data_train_loader = DataLoader(dataset_feature_train,
                                                   shuffle=True,
                                                   batch_size=args.batch_size,
                                                   collate_fn=collate_fn)

                    dataset_feature_test = GraphFeatureExtractor(graphs_test,
                                                                 max_dataset_nodes=max_dataset_node_num,
                                                                 max_graph_nodes=max_graph_nodes_num,
                                                                 node_attr_dim=node_attr_dim,
                                                                 transform=Compose(
                                                                     [ShortestPathGenerator(), OneHotEdgeAttr()]),
                                                                 )
                    data_test_loader = DataLoader(dataset_feature_test,
                                                  shuffle=False,
                                                  batch_size=1,
                                                  collate_fn=collate_fn)
                    result = train(data_train_loader, data_test_loader, model_teacher, model_student, args)

                    result_auc_list.append(result[0])
                    result_precision_list.append(result[1])
                    result_recall_list.append(result[2])
                    result_f1_list.append(result[3])

                record_result(data_set, result_auc_list, result_precision_list, result_recall_list, result_f1_list)
