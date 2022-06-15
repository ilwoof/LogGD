#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 5/05/2022 1:04 pm

@author : Yongzheng Xie
@email : ilwoof@gmail.com
"""
import numpy as np

def record_result(data_set, result_auc_list, result_precision_list, result_recall_list, result_f1_list):

    result_auc = np.array(result_auc_list)
    auc_avg = np.mean(result_auc_list)
    auc_std = np.std(result_auc_list)

    result_precision = np.array(result_precision_list)
    precision_avg = np.mean(result_precision_list)
    precision_std = np.std(result_precision_list)

    result_recall = np.array(result_recall_list)
    recall_avg = np.mean(result_recall_list)
    recall_std = np.std(result_recall_list)

    result_f1 = np.array(result_f1_list)
    f1_avg = np.mean(result_f1_list)
    f1_std = np.std(result_f1_list)
    with open('Experiment_results.txt', 'a+') as f:
        print(f"The detection result for {data_set}:\n"
              f'{"-" * 100}\n'
              f'auroc average: {auc_avg}, std: {auc_std}\n'
              f'auroc: {result_auc}\n'
              f'{"-" * 100}\n'
              f'precision average: {precision_avg}, std: {precision_std}\n'
              f'precision: {result_precision}\n'
              f'{"-"*100}\n'
              f'recall average: {recall_avg}, std: {recall_std}\n'
              f'recall: {result_recall}\n'
              f'{"-"*100}\n'
              f'f1 average: {f1_avg}, std: {f1_std}\n'
              f'f1: {result_f1}\n'
              , file=f)
    print(f"The detection result for {data_set}:\n"
          f'{"-" * 100}\n'
          f'auroc average: {auc_avg}, std: {auc_std}\n'
          f'auroc: {result_auc}\n'
          f'{"-" * 100}\n'
          f'precision average: {precision_avg}, std: {precision_std}\n'
          f'precision: {result_precision}\n'
          f'{"-" * 100}\n'
          f'recall average: {recall_avg}, std: {recall_std}\n'
          f'recall: {result_recall}\n'
          f'{"-" * 100}\n'
          f'f1 average: {f1_avg}, std: {f1_std}\n'
          f'f1: {result_f1}\n'
          )
