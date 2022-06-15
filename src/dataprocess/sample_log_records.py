#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 26/05/2022 8:13 pm

@author : Yongzheng Xie
@email : ilwoof@gmail.com
"""

from itertools import islice

def sampling_topN_records(input_file, output_file, top_num=5000000):
    with open(output_file, "w") as out_f:
        with open(input_file, 'r') as in_f:
            out_f.writelines(islice(in_f, top_num))


if __name__ == '__main__':
    data_path = f"../logs/"
    for dataset in ['Spirit', 'Thunderbird']:
        if dataset == 'Spirit':
            volume = '1G'
        else:
            assert dataset == 'Thunderbird'
            volume = '10M'
        in_file = f"{data_path}/{dataset}/{dataset}{volume}.log"
        out_file = f"{data_path}/{dataset}/{dataset}5M.log"
        sampling_topN_records(in_file, out_file)
        print(f"The sampled log file {dataset}5M.log has been generated")
