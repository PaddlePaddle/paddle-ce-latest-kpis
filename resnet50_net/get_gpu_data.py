#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
import os
import sys
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
parser.add_argument(
    '--reduce_strategy',
    type=str,
    default='AllReduce',
    choices=['AllReduce', 'Reduce'],
    help='The reduce strategy.')
args = parser.parse_args()


def save_gpu_data():
    cards = os.environ.get('CUDA_VISIBLE_DEVICES')
    cards = str(len(cards.split(",")))
    if int(cards) > 1:
        run_info = args.reduce_strategy + "_" + cards + "_Cards"
    else:
        run_info = cards + "_Cards"

    mem_list = []
    with open('memory.txt', 'r') as f:
        for i, data in enumerate(f.readlines()):
            if i == 0:
                continue
            mem_list.append(int(data.split("\n")[0].split(" ")[0]))

    kpi_name = '%s_%s_%s_gpu_memory' % (args.data_set, args.batch_size,
                                        run_info)
    gpu_memory_kpi = None
    for kpi in tracking_kpis:
        if kpi.name == kpi_name:
            gpu_memory_kpi = kpi
    assert gpu_memory_kpi is not None, kpi_nam + "is not found."
    gpu_memory_kpi.add_record(max(mem_list))
    gpu_memory_kpi.persist()


if __name__ == "__main__":
    save_gpu_data()
