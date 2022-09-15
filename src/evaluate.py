#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 5/05/2022 1:04 pm

@author : Yongzheng Xie
@email : ilwoof@gmail.com
"""
import os
import numpy as np
from datetime import datetime

RST_HEADER = np.array(['run_time', 'model_name', 'data_set', 'window_size',
                       'embedding_type', 'epochs', 'patience', 'no_validation',
                       'max_hops', 'test_ratio', 'sampling_ratio', 'anomaly_ratio',
                       'dataset_nodes', 'max_graph_nodes',
                       'average_precision', 'std_precision',
                       'average_recall', 'std_recall',
                       'average_f1', 'std_f1',
                       'average_roc', 'std_roc'])

def gather_results(result_dict, one_result):
    result_dict['roc'].append(one_result['roc'])
    result_dict['precision'].append(one_result['precision'])
    result_dict['recall'].append(one_result['recall'])
    result_dict['f1'].append(one_result['f1_score'])
    return result_dict


def format_output(data_dict, newline_pos=15):
    assert isinstance(data_dict, dict)
    str_list = []
    for i, (k, v) in enumerate(data_dict.items()):
        if (i + 1) % newline_pos == 0:
            str_list.append(f"{k}={v}\n")
        else:
            str_list.append(f"{k}={v}")
    return ', '.join(str_list)


def record_result(result_file, data_config, result_dict):
    results = record_data_info(data_config, result_dict)
    save_result(result_file, results, RST_HEADER)
    log_content = f"{'-' * 100}\n" \
                  f"The detection result on dataset {data_config['data_set']} at window_size={data_config['window_size']}:\n" \
                  f"{'-' * 100}\n" \
                  f"precision average: {np.array(result_dict['precision']).mean():.5f}, std: {np.array(result_dict['precision']).std():.5f}\n" \
                  f"precision: {np.array(result_dict['precision']).round(5)}\n" \
                  f"{'-' * 100}\n" \
                  f"recall average: {np.array(result_dict['recall']).mean():.5f}, std: {np.array(result_dict['recall']).std():.5f}\n" \
                  f"recall: {np.array(result_dict['recall']).round(5)}\n" \
                  f"{'-' * 100}\n" \
                  f"f1 average: {np.array(result_dict['f1']).mean():.5f}, std: {np.array(result_dict['f1']).std():.5f}\n" \
                  f"f1: {np.array(result_dict['f1']).round(5)}\n" \
                  f"{'-' * 100}\n" \
                  f"aucroc average: {np.array(result_dict['roc']).mean():.5f}, std: {np.array(result_dict['roc']).std():.5f}\n" \
                  f"aucroc: {np.array(result_dict['roc']).round(5)}\n"
    print(log_content)


def record_data_info(data_setting, result_dict):
    rst_all = []
    for item_name in RST_HEADER:
        if item_name in list(data_setting.keys()):
            rst_all = record_result_values(rst_all, item_name, data_setting[item_name])
    rst_all = record_result_values(rst_all, 'run_time', datetime.now().strftime("%Y/%m/%d-%H:%M:%S"))
    rst_all = record_result_values(rst_all, 'average_precision', f"{np.array(result_dict['precision']).mean():.5f}")
    rst_all = record_result_values(rst_all, 'std_precision', f"{np.array(result_dict['precision']).std():.5f}")
    rst_all = record_result_values(rst_all, 'average_recall', f"{np.array(result_dict['recall']).mean():.5f}")
    rst_all = record_result_values(rst_all, 'std_recall', f"{np.array(result_dict['recall']).std():.5f}")
    rst_all = record_result_values(rst_all, 'average_f1', f"{np.array(result_dict['f1']).mean():.5f}")
    rst_all = record_result_values(rst_all, 'std_f1', f"{np.array(result_dict['f1']).std():.5f}")
    rst_all = record_result_values(rst_all, 'average_roc', f"{np.array(result_dict['roc']).mean():.5f}")
    rst_all = record_result_values(rst_all, 'std_roc', f"{np.array(result_dict['roc']).std():.5f}")
    return rst_all

def record_result_values(rst_all, item, item_value):
    if item not in RST_HEADER:
        print(f'Wrong item({item}), valid item: {RST_HEADER}')
        return
    idx = np.where(RST_HEADER == item)[0][0]
    if len(rst_all) == 0:
        rst_one = ['-' for i in range(len(RST_HEADER))]
        rst_one[idx] = item_value
        rst_all.append(rst_one)
    else:
        rst_all[len(rst_all) - 1][idx] = item_value
    return rst_all

def save_result(filename, rst_all, header, append=True):
    if os.path.exists(filename):
        if not append:
            f = open(filename, 'wb')
            header_need = True
        else:
            f = open(filename, 'ab')
            header_need = False

    else:
        f = open(filename, 'xb')
        header_need = True

    if header_need:
        header = ','.join(header)
        np.savetxt(f, rst_all, delimiter=',', fmt='%s', header=header, comments='')
    else:
        np.savetxt(f, rst_all, delimiter=',', fmt='%s')
    f.close()
    return
