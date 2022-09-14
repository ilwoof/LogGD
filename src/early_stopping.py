#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 19/05/2022 12:41 pm

@author : Yongzheng Xie
@email : ilwoof@gmail.com
"""
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.score_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.loss = np.Inf
        self.result_dict = False
        self.precision = 0
        self.recall = 0
        self.roc_auc = 0
        self.f1_score = 0

    def __call__(self, result, model, monitor_metric='loss'):
        """
        Args:
            result (dict or float): {'loss': xxx.xxxx, 'precision': xxx.xxxx,
                                     'recall': xxx.xxxx, 'f1_score': xxx.xxxx, 'roc': xxx.xxxx}
            monitor_metric (str): choice 'loss', 'f1_score', default: 'loss'
        """
        assert monitor_metric in ['loss', 'f1_score']

        if isinstance(result, dict):
            self.result_dict = True
            if monitor_metric == 'loss':
                score = -result['loss']
            else:
                score = result['f1_score']
        else:
            self.result_dict = False
            if monitor_metric == 'loss':
                score = -result
            else:
                score = result

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, metric=monitor_metric)

            if isinstance(result, dict):
                self.loss = result['loss']
                self.precision = result['precision']
                self.recall = result['recall']
                self.f1_score = result['f1_score']
                self.roc = result['roc']
            else:
                if monitor_metric == 'loss':
                    self.loss = result
                else:
                    self.f1_score = result

        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping")
        else:
            self.best_score = score
            self.save_checkpoint(score, model, metric=monitor_metric)
            self.counter = 0

            if isinstance(result, dict):
                self.loss = result['loss']
                self.precision = result['precision']
                self.recall = result['recall']
                self.f1_score = result['f1_score']
                self.roc = result['roc']
            else:
                if monitor_metric == 'loss':
                    self.loss = result
                else:
                    self.f1_score = result

        return self.early_stop

    def save_checkpoint(self, score, model, metric):
        '''Saves model when validation loss decrease.'''

        if self.verbose:
            self.trace_func(f'{metric} ({abs(self.score_min):.6f} --> {abs(score):.6f})')
        torch.save(model.state_dict(), self.path)
        self.score_min = abs(score)

    def get_best_result(self):
        '''Return the best metric for the model'''
        if self.result_dict:
            best_result = {'loss': self.loss, 'f1_score': self.f1_score, 'precision': self.precision,
                           'recall': self.recall, 'roc': self.roc}
        else:
            best_result = abs(self.best_score)
        return best_result
