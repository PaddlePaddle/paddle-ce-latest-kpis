# this file is only used for continuous evaluation test!

import os
import sys

sys.path.append(os.environ['ceroot'])
from kpi import CostKpi
from kpi import DurationKpi
from kpi import AccKpi

cdssmNet_each_pass_duration_card1_kpi = DurationKpi('cdssmNet_each_pass_duration_card1', 0.03, 0, actived=True)
cdssmNet_train_avg_cost_card1_kpi = CostKpi('cdssmNet_train_avg_cost_card1', 0.02, 0, actived=True)
cdssmNet_train_avg_acc_card1_kpi = AccKpi('cdssmNet_train_avg_acc_card1', 0.01, 0, actived=True)

DecAttNet_each_pass_duration_card1_kpi = DurationKpi('DecAttNet_each_pass_duration_card1', 0.04, 0, actived=True)
DecAttNet_train_avg_cost_card1_kpi = CostKpi('DecAttNet_train_avg_cost_card1', 0.03, 0, actived=True)
DecAttNet_train_avg_acc_card1_kpi = AccKpi('DecAttNet_train_avg_acc_card1', 0.02, 0, actived=True)

InferSentNet_v1_each_pass_duration_card1_kpi = DurationKpi('InferSentNet_v1_each_pass_duration_card1', 0.01, 0,
                                                           actived=True)
InferSentNet_v1_train_avg_cost_card1_kpi = CostKpi('InferSentNet_v1_train_avg_cost_card1', 0.01, 0, actived=True)
InferSentNet_v1_train_avg_acc_card1_kpi = AccKpi('InferSentNet_v1_train_avg_acc_card1', 0.01, 0, actived=True)

InferSentNet_v2_each_pass_duration_card1_kpi = DurationKpi('InferSentNet_v2_each_pass_duration_card1', 0.01, 0,
                                                           actived=True)
InferSentNet_v2_train_avg_cost_card1_kpi = CostKpi('InferSentNet_v2_train_avg_cost_card1', 0.01, 0, actived=True)
InferSentNet_v2_train_avg_acc_card1_kpi = AccKpi('InferSentNet_v2_train_avg_acc_card1', 0.01, 0, actived=True)

SSENet_each_pass_duration_card1_kpi = DurationKpi('SSENet_each_pass_duration_card1', 0.01, 0, actived=True)
SSENet_train_avg_cost_card1_kpi = CostKpi('SSENet_train_avg_cost_card1', 0.01, 0, actived=True)
SSENet_train_avg_acc_card1_kpi = AccKpi('SSENet_train_avg_acc_card1', 0.01, 0, actived=True)

tracking_kpis = [
    cdssmNet_each_pass_duration_card1_kpi,
    cdssmNet_train_avg_cost_card1_kpi,
    cdssmNet_train_avg_acc_card1_kpi,
    DecAttNet_each_pass_duration_card1_kpi,
    DecAttNet_train_avg_cost_card1_kpi,
    DecAttNet_train_avg_acc_card1_kpi,
    InferSentNet_v1_each_pass_duration_card1_kpi,
    InferSentNet_v1_train_avg_cost_card1_kpi,
    InferSentNet_v1_train_avg_acc_card1_kpi,
    InferSentNet_v2_each_pass_duration_card1_kpi,
    InferSentNet_v2_train_avg_cost_card1_kpi,
    InferSentNet_v2_train_avg_acc_card1_kpi,
    SSENet_each_pass_duration_card1_kpi,
    SSENet_train_avg_cost_card1_kpi,
    SSENet_train_avg_acc_card1_kpi
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
