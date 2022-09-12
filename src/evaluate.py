#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 5/05/2022 1:04 pm

@author : Yongzheng Xie
@email : ilwoof@gmail.com
"""

import numpy as np

def log_info(f_result, str_content, sep_flag=None):
    assert sep_flag in ['-', '*'] or (sep_flag is None)
    if sep_flag is not None:
        print(sep_flag * 100)
    print(str_content)

    with open(f_result, 'a+') as f:
        if sep_flag is not None:
            print(sep_flag * 100, file=f)
        print(str_content, file=f)


def record_result(result_file, data_set, window_size, result_dict):

    log_content = f"{'-' * 100}\n" \
                  f"The detection result on dataset {data_set} at window_size={window_size}:\n" \
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
                  f"aucroc average: {np.array(result_dict['auc']).mean():.5f}, std: {np.array(result_dict['auc']).std():.5f}\n" \
                  f"aucroc: {np.array(result_dict['auc']).round(5)}\n" \

    log_info(result_file, log_content)
