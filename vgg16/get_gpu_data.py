#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
"""
File: get_gpu_data.py
Author: paddle(paddle@baidu.com)
Date: 2018/04/02 15:57:14
"""
import argparse
from continuous_evaluation import tracking_kpis

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--batch_size', type=int, default=128, help="Batch size for training.")
parser.add_argument(
    '--data_set',
    type=str,
    default='cifar10',
    choices=['cifar10', 'flowers'],
    help='Optional dataset for benchmark.')
args = parser.parse_args()


def save_gpu_data():
    mem_list = []
    with open('memory.txt', 'r') as f:
        for i, data in enumerate(f.readlines()):
            if i == 0:
                continue
            mem_list.append(int(data.split("\n")[0].split(" ")[0]))
    gpu_memory_factor = None
    for kpi in tracking_kpis:
        if kpi.name == '%s_%s_gpu_memory' % (args.data_set, args.batch_size):
            gpu_memory_kpi = kpi
    gpu_memory_kpi.add_record(max(mem_list))
    gpu_memory_kpi.persist()


if __name__ == "__main__":
    save_gpu_data()
