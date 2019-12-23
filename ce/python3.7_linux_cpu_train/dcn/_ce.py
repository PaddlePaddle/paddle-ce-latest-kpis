# this file is only used for continuous evaluation test!

import os
import sys

sys.path.append(os.environ['ceroot'])
from kpi import CostKpi
from kpi import DurationKpi
from kpi import AccKpi

# NOTE kpi.py should shared in models in some way!!!!

test_auc_cpu1_thread1_kpi = AccKpi(
    'test_auc_cpu1_thread1', 0.01, 0, actived=True, desc='test_auc')
test_auc_cpu1_thread20_kpi = AccKpi(
    'test_auc_cpu1_thread20', 0.01, 0, actived=True, desc='test_auc')
test_auc_cpu5_thread20_kpi = AccKpi(
    'test_auc_cpu5_thread20', 0.04, 0, actived=True, desc='test_auc')
tracking_kpis = [test_auc_cpu1_thread1_kpi,
                 test_auc_cpu1_thread20_kpi,
                 test_auc_cpu5_thread20_kpi]


def parse_log(log):
    '''
    This method should be implemented by model developers.

    The suggestion:

    each line in the log should be key, value, for example:

    "
    kpis\ttest_auc\t1.0
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
