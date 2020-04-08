# this file is only used for continuous evaluation test!

import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi

AttentionCluster_loss_kpi = CostKpi(
    'AttentionCluster_loss', 0.08, 0, actived=True, desc='train cost')
AttentionCluster_time_kpi = DurationKpi(
    'AttentionCluster_time',
    0.08,
    0,
    actived=True,
    desc='train time')
AttentionLSTM_loss_kpi = CostKpi(
    'AttentionLSTM_loss', 0.1, 0, actived=True, desc='train cost')
AttentionLSTM_time_kpi = DurationKpi(
    'AttentionLSTM_time',
    0.08,
    0,
    actived=True,
    desc='train time')
NEXTVLAD_loss_kpi = CostKpi(
    'NEXTVLAD_loss', 0.08, 0, actived=True, desc='train cost')
NEXTVLAD_time_kpi = DurationKpi(
    'NEXTVLAD_time',
    0.08,
    0,
    actived=True,
    desc='train time')
STNET_loss_kpi = CostKpi(
    'STNET_loss', 0.08, 0, actived=True, desc='train cost')
STNET_time_kpi = DurationKpi(
    'STNET_time',
    0.08,
    0,
    actived=True,
    desc='train time')
TSM_loss_kpi = CostKpi(
    'TSM_loss', 0.08, 0, actived=True, desc='train cost')
TSM_time_kpi = DurationKpi(
    'TSM_time',
    0.08,
    0,
    actived=True,
    desc='train time')
TSN_loss_kpi = CostKpi(
    'TSN_loss', 0.08, 0, actived=True, desc='train cost')
TSN_time_kpi = DurationKpi(
    'TSN_time',
    0.08,
    0,
    actived=True,
    desc='train time')
NONLOCAL_loss_kpi = CostKpi(
    'NONLOCAL_loss', 0.08, 0, actived=True, desc='train cost')
NONLOCAL_time_kpi = DurationKpi(
    'NONLOCAL_time',
    0.08,
    0,
    actived=True,
    desc='train time')
BMN_loss_kpi = CostKpi(
    'BMN_loss', 0.08, 0, actived=True, desc='train cost')
BMN_time_kpi = DurationKpi(
    'BMN_time',
    0.08,
    0,
    actived=True,
    desc='train time')
BsnTem_loss_kpi = CostKpi(
    'BsnTem_loss', 0.08, 0, actived=True, desc='train cost')
BsnTem_time_kpi = DurationKpi(
    'BsnTem_time',
    0.08,
    0,
    actived=True,
    desc='train time')
ETS_loss_kpi = CostKpi(
    'ETS_loss', 0.08, 0, actived=True, desc='train cost')
ETS_time_kpi = DurationKpi(
    'ETS_time',
    0.08,
    0,
    actived=True,
    desc='train time')
TALL_loss_kpi = CostKpi(
    'TALL_loss', 0.08, 0, actived=True, desc='train cost')
TALL_time_kpi = DurationKpi(
    'TALL_time',
    0.08,
    0,
    actived=True,
    desc='train time')

tracking_kpis = [
    AttentionCluster_loss_kpi, AttentionCluster_time_kpi,
    AttentionLSTM_loss_kpi, AttentionLSTM_time_kpi,
    NEXTVLAD_loss_kpi, NEXTVLAD_time_kpi,
    STNET_loss_kpi, STNET_time_kpi, TSM_loss_kpi, TSM_time_kpi, 
    TSN_loss_kpi, TSN_time_kpi, NONLOCAL_loss_kpi, NONLOCAL_time_kpi,
    BMN_loss_kpi, BMN_time_kpi, BsnTem_loss_kpi, BsnTem_time_kpi, 
    ETS_loss_kpi, ETS_time_kpi, TALL_loss_kpi, TALL_time_kpi
]


def parse_log(log):
    '''
    This method should be implemented by model developers.
    The suggestion:
    each line in the log should be key, value, for example:
    "
    tsm_loss\t1.0
    tsm_time\t1.0
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
