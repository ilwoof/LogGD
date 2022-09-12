#!/usr/bin/env python

import sys
import os
import pandas as pd
sys.path.append('../../../')
from src.logparser.Drain import Drain
from src.logparser.utils import evaluator

input_dir = '../../../dataset/raw/'  # The input directory of log file
output_dir = 'Drain_result/'  # The output directory of parsing results

benchmark_settings = {
    # 'HDFS': {
    #     'log_file': 'HDFS/HDFS.log',
    #     'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    #     'st': 0.5,
    #     'depth': 4
    #     },

    # 'BGL': {
    #     'log_file': 'BGL/BGL.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    #     'st': 0.5,
    #     'depth': 4
    # },
    'BGL': {
        'log_file': 'bgl/BGL.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+', r'(\/.*?\.[\S:]+)'],
        'st': 0.5,
        'depth': 4
    },

    #  #  The following configuration is wrong, which lead to many extra templates produced
    # 'Spirit': {
    #     'log_file': 'Spirit/Spirit5M.log',
    #     'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Content>',
    #     'regex': [],
    #     'st': 0.5,
    #     'depth': 4
    #     },
    # 'Thunderbird': {
    #     'log_file': 'Thunderbird/Thunderbird5M.log',
    #     'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
    #     'regex': [r'(\d+\.){3}\d+'],
    #     'st': 0.5,
    #     'depth': 4
    #     },

    'Spirit': {
        'log_file': 'spirit/Spirit1G.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Content>',
        'regex': [r'(\d+\.){3}\d+', r'(\/.*?\.[\S:]+)', r'(/[a-zA-Z0-9_-]+)+',
                  r'((?<=[^A-Za-z0-9])|^)([\-\+]?\d+)((?=[^A-Za-z0-9])|$)',
                  r'((?<=[^A-Za-z0-9])|^)(0x[a-f0-9A-F]+)((?=[^A-Za-z0-9])|$)'],
        'st': 0.5,
        'depth': 4
    },
    'Thunderbird': {
        'log_file': 'tbd/Thunderbird10M.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'(\/.*?\.[\S:]+)', r'(/[a-zA-Z0-9_-]+)+',
                  r'((?<=[^A-Za-z0-9])|^)([\-\+]?\d+)((?=[^A-Za-z0-9])|$)',
                  r'((?<=[^A-Za-z0-9])|^)(0x[a-f0-9A-F]+)((?=[^A-Za-z0-9])|$)'],
        'st': 0.5,
        'depth': 4
    },
}

benchmark_result = []
for dataset, setting in benchmark_settings.items():
    print('\n=== Evaluation on %s ===' % dataset)
    indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
    log_file = os.path.basename(setting['log_file'])

    parser = Drain.LogParser(log_format=setting['log_format'], indir=indir, outdir=output_dir, rex=setting['regex'],
                             depth=setting['depth'], st=setting['st'])
    parser.parse(log_file)

    # F1_measure, accuracy = evaluator.evaluate(
    #                        groundtruth=os.path.join(indir, log_file + '_structured.csv'),
    #                        parsedresult=os.path.join(output_dir, log_file + '_structured.csv')
    #                        )
    # benchmark_result.append([dataset, F1_measure, accuracy])
#
#
# print('\n=== Overall evaluation results ===')
# df_result = pd.DataFrame(benchmark_result, columns=['Dataset', 'F1_measure', 'Accuracy'])
# df_result.set_index('Dataset', inplace=True)
# print(df_result)
# df_result.T.to_csv('Drain_benchmark_result.csv')
