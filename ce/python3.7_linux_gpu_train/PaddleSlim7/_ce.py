# this file is only used for continuous evaluation test!

import os
import sys

sys.path.append(os.environ['ceroot'])
from kpi import CostKpi
from kpi import DurationKpi
from kpi import AccKpi

# NOTE kpi.py should shared in models in some way!!!!
slim_uniform_prune_test_acc_top1_kpi = AccKpi(
    'slim_uniform_prune_test_acc_top1', 0.02, 0, actived=True, desc=' uniform_prune TOP1 ACC')
slim_uniform_prune_test_acc_top5_kpi = AccKpi(
    'slim_uniform_prune_test_acc_top5', 0.02, 0, actived=True, desc=' uniform_prune TOP5 ACC')

slim_sen_prune_test_acc_top1_kpi = AccKpi(
    'slim_sen_prune_test_acc_top1', 0.06, 0, actived=True, desc='sen_prune TOP1 ACC')
slim_sen_prune_test_acc_top5_kpi = AccKpi(
    'slim_sen_prune_test_acc_top5', 0.04, 0, actived=True, desc='sen_prune TOP5 ACC')

slim_quan_v1_test_acc_top1_kpi = AccKpi(
    'slim_quan_v1_test_acc_top1', 0.06, 0, actived=True, desc='slim_quan_v1 TOP1 ACC')
slim_quan_v1_test_acc_top5_kpi = AccKpi(
    'slim_quan_v1_test_acc_top5', 0.04, 0, actived=True, desc='slim_quan_v1 TOP5 ACC')

slim_dist_v1_test_acc_top1_kpi = AccKpi(
    'slim_dist_v1_test_acc_top1', 0.06, 0, actived=True, desc='slim_dist_v1 TOP1 ACC')
slim_dist_v1_test_acc_top5_kpi = AccKpi(
    'slim_dist_v1_test_acc_top5', 0.06, 0, actived=True, desc='slim_dist_v1 TOP5 ACC')

class_prune_v1_test_acc_top1_kpi = AccKpi(
    'class_prune_v1_test_acc_top1', 0.05, 0, actived=True, desc='prune_v1 TOP1 ACC')
class_prune_v1_test_acc_top5_kpi = AccKpi(
    'class_prune_v1_test_acc_top5', 0.03, 0, actived=True, desc='prune_v1 TOP5 ACC')

class_quan_v2_test_acc_top1_kpi = AccKpi(
    'class_quan_v2_test_acc_top1', 0.01, 0, actived=True, desc='quan_v2 TOP1 ACC')
class_quan_v2_test_acc_top5_kpi = AccKpi(
    'class_quan_v2_test_acc_top5', 0.01, 0, actived=True, desc='quan_v2 TOP5 ACC')

class_dist_resnet34_test_acc_top1_kpi = AccKpi(
    'class_dist_resnet34_test_acc_top1', 0.06, 0, actived=True, desc='dist_resnet34 TOP1 ACC')
class_dist_resnet34_test_acc_top5_kpi = AccKpi(
    'class_dist_resnet34_test_acc_top5', 0.06, 0, actived=True, desc='dist_resnet34 TOP5 ACC')

tracking_kpis = [slim_uniform_prune_test_acc_top1_kpi,
                 slim_uniform_prune_test_acc_top5_kpi,
                 slim_sen_prune_test_acc_top1_kpi,
                 slim_sen_prune_test_acc_top5_kpi,
                 slim_quan_v1_test_acc_top1_kpi,
                 slim_quan_v1_test_acc_top5_kpi,
                 slim_dist_v1_test_acc_top1_kpi,
                 slim_dist_v1_test_acc_top5_kpi,
                 class_prune_v1_test_acc_top1_kpi,
                 class_prune_v1_test_acc_top5_kpi,
                 class_quan_v2_test_acc_top1_kpi,
                 class_quan_v2_test_acc_top5_kpi,
                 class_dist_resnet34_test_acc_top1_kpi,
                 class_dist_resnet34_test_acc_top5_kpi
                 ]


def parse_log(log):
    '''
    This method should be implemented by model developers.

    The suggestion:

    each line in the log should be key, value, for example:

    "
    kpis\ttest_acc_top1\t1.0
    kpis\ttest_acc_top5\t1.0
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
