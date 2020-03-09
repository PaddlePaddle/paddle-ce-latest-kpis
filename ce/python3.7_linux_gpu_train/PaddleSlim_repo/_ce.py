# this file is only used for continuous evaluation test!

import os
import sys

sys.path.append(os.environ['ceroot'])
from kpi import CostKpi
from kpi import DurationKpi
from kpi import AccKpi

dist_acc_top1_gpu1_kpi = AccKpi('dist_acc_top1_gpu1', 0.04, 0, actived=True)
dist_acc_top5_gpu1_kpi = AccKpi('dist_acc_top5_gpu1', 0.03, 0, actived=True)
dist_acc_top1_gpu8_kpi = AccKpi('dist_acc_top1_gpu8', 0.07, 0, actived=True)
dist_acc_top5_gpu8_kpi = AccKpi('dist_acc_top5_gpu8', 0.04, 0, actived=True)

quant_aware_acc_top1_gpu1_kpi = AccKpi('quant_aware_acc_top1_gpu1', 0.02, 0, actived=True)
quant_aware_acc_top5_gpu1_kpi = AccKpi('quant_aware_acc_top5_gpu1', 0.02, 0, actived=True)
quant_aware_acc_top1_gpu8_kpi = AccKpi('quant_aware_acc_top1_gpu8', 0.07, 0, actived=True)
quant_aware_acc_top5_gpu8_kpi = AccKpi('quant_aware_acc_top5_gpu8', 0.06, 0, actived=True)

prune_v1_acc_top1_gpu1_kpi = AccKpi('prune_v1_acc_top1_gpu1', 0.02, 0, actived=True)
prune_v1_acc_top5_gpu1_kpi = AccKpi('prune_v1_acc_top5_gpu1', 0.01, 0, actived=True)
prune_v1_acc_top1_gpu8_kpi = AccKpi('prune_v1_acc_top1_gpu8', 0.01, 0, actived=True)
prune_v1_acc_top5_gpu8_kpi = AccKpi('prune_v1_acc_top5_gpu8', 0.01, 0, actived=True)

tracking_kpis = [
    dist_acc_top1_gpu1_kpi,
    dist_acc_top5_gpu1_kpi,
    dist_acc_top1_gpu8_kpi,
    dist_acc_top5_gpu8_kpi,
    quant_aware_acc_top1_gpu1_kpi,
    quant_aware_acc_top5_gpu1_kpi,
    quant_aware_acc_top1_gpu8_kpi,
    quant_aware_acc_top5_gpu8_kpi,
    prune_v1_acc_top1_gpu1_kpi,
    prune_v1_acc_top5_gpu1_kpi,
    prune_v1_acc_top1_gpu8_kpi,
    prune_v1_acc_top5_gpu8_kpi
]


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
    kpi_tracker = {}
    for kpi in tracking_kpis:
        kpi_tracker[kpi.name] = kpi

    for (kpi_name, kpi_value) in parse_log(log):
        print(kpi_name, kpi_value)
        kpi_tracker[kpi_name].add_record(kpi_value)
        kpi_tracker[kpi_name].persist()


if __name__ == '__main__':
    log = sys.stdin.read()
    log_to_ce(log)
