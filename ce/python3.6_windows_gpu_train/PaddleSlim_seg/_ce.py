#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import re
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi
from kpi import AccKpi

distill_train_loss_kpi = CostKpi(
    'distill_train_loss', 0.05, 0, actived=True, desc='distill_train_loss')
# distillation_eval_loss_kpi = CostKpi(
#     'distill_eval_loss', 0.05, 0, actived=True, desc='distill_eval_loss')

prune_train_loss_kpi = CostKpi(
    'prune_train_loss', 0.05, 0, actived=True, desc='prune_loss')
    
quant_train_loss_kpi = CostKpi(
    'quant_train_loss', 0.05, 0, actived=True, desc='quant_train_loss')
quant_eval_loss_kpi = CostKpi(
    'quant_eval_loss', 0.05, 0, actived=True, desc='quant_eval_loss')   


nas_train_loss_kpi = CostKpi(
    'nas_train_loss', 0.05, 0, actived=True, desc='nas_train_loss')
nas_eval_loss_kpi = CostKpi(
    'nas_eval_loss', 0.05, 0, actived=True, desc='nas_eval_loss')
    
tracking_kpis = [distill_train_loss_kpi, prune_train_loss_kpi, quant_train_loss_kpi, quant_eval_loss_kpi, prune_train_loss_kpi, nas_train_loss_kpi, nas_eval_loss_kpi] 


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


