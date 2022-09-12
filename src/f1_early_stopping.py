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
        self.f1_score_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_threshold = 0.5
        self.precision = 0
        self.recall = 0
        self.roc = 0

    def __call__(self, result, model, threshold=0.5):

        score = result['f1_score']

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(result['f1_score'], model)
            self.precision = result['precision']
            self.recall = result['recall']
            self.roc = result['roc']

        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping")
        else:
            self.best_score = score
            self.save_checkpoint(result['f1_score'], model)
            self.counter = 0
            self.best_threshold = threshold
            self.precision = result['precision']
            self.recall = result['recall']
            self.roc = result['roc']

        return {'early_stop': self.early_stop, 'precision': self.precision, 'recall': self.recall, 'f1_score': self.best_score, 'roc': self.roc}

    def save_checkpoint(self, f1_score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'F1 score ({self.f1_score_min:.6f} --> {f1_score:.6f})')
        torch.save(model.state_dict(), self.path)
        self.f1_score_min = f1_score

    def get_threshold(self):
        '''Return best threshold of the prediction value.'''
        return self.best_threshold
