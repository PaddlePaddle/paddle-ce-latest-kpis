#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import re
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi
from kpi import AccKpi

distillation_train_loss_kpi = CostKpi(
    'distillation_train_loss', 0.05, 0, actived=True, desc='train_loss')
distillation_valid_loss_kpi = CostKpi(
    'distillation_valid_loss', 0.05, 0, actived=True, desc='valid_loss')
tracking_kpis = [distillation_train_loss_kpi, distillation_valid_loss_kpi] 

quant_aware_train_loss_kpi = CostKpi(
    'quant_aware_train_loss', 0.05, 0, actived=True, desc='quant_aware_train_loss')
    
prune_loss_kpi = CostKpi(
    'prune_loss', 0.05, 0, actived=True, desc='prune_loss')
    
nan_test_loss_kpi = CostKpi(
    'nan_test_loss', 0.05, 0, actived=True, desc='test_loss')
nas_current_flops_kpi = AccKpi(
    'nas_current_flops', 0.05, 0, actived=True, desc='current_flops')
    
tracking_kpis = [distillation_train_loss_kpi, distillation_valid_loss_kpi, quant_aware_train_loss_kpi, prune_loss_kpi, nan_test_loss_kpi, nas_current_flops_kpi] 


def parse_log(log):
    '''
    This method should be implemented by model developers.
    The suggestion:
    each line in the log should be key, value, for example:
    "
    train_cost\t1.0
    test_cost\t1.0
    train_cost\t1.0
    train_cost\t1.0
    train_acc\t1.2
    "
    '''
    for line in log.split('\n'):
        fs = line.strip().split('\t')
        print(fs)
        if len(fs) == 3 and fs[0] == 'kpis':
            kpi_name = fs[1]
            kpi_value = float(fs[2])
            yield kpi_name, kpi_value


def log_to_ce(log):
    """
    log to ce
    """
    kpi_tracker = {}
    for kpi in tracking_kpis:
        kpi_tracker[kpi.name] = kpi

    for(kpi_name, kpi_value) in parse_log(log):
        kpi_tracker[kpi_name].add_record(kpi_value)
        kpi_tracker[kpi_name].persist()


if __name__ == '__main__':
    log = sys.stdin.read()
    log_to_ce(log)


