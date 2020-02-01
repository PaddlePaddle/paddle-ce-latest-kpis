# this file is only used for continuous evaluation test!

import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi

AttentionCluster_loss_card1_kpi = CostKpi(
    'AttentionCluster_loss_card1', 0.08, 0, actived=True, desc='train cost')
AttentionCluster_time_card1_kpi = DurationKpi(
    'AttentionCluster_time_card1',
    0.08,
    0,
    actived=True,
    desc='train speed in one GPU card')
AttentionCluster_loss_card8_kpi = CostKpi(
    'AttentionCluster_loss_card8', 0.08, 0, actived=True, desc='train cost')
AttentionCluster_time_card8_kpi = DurationKpi(
    'AttentionCluster_time_card8',
    0.08,
    0,
    actived=True,
    desc='train time in four GPU card')
AttentionLSTM_loss_card1_kpi = CostKpi(
    'AttentionLSTM_loss_card1', 0.1, 0, actived=True, desc='train cost')
AttentionLSTM_time_card1_kpi = DurationKpi(
    'AttentionLSTM_time_card1',
    0.08,
    0,
    actived=True,
    desc='train speed in one GPU card')
AttentionLSTM_loss_card8_kpi = CostKpi(
    'AttentionLSTM_loss_card8', 0.08, 0, actived=True, desc='train cost')
AttentionLSTM_time_card8_kpi = DurationKpi(
    'AttentionLSTM_time_card8',
    0.08,
    0,
    actived=True,
    desc='train time in four GPU card')
NEXTVLAD_loss_card1_kpi = CostKpi(
    'NEXTVLAD_loss_card1', 0.08, 0, actived=True, desc='train cost')
NEXTVLAD_time_card1_kpi = DurationKpi(
    'NEXTVLAD_time_card1',
    0.08,
    0,
    actived=True,
    desc='train speed in one GPU card')
NEXTVLAD_loss_card8_kpi = CostKpi(
    'NEXTVLAD_loss_card8', 0.08, 0, actived=True, desc='train cost')
NEXTVLAD_time_card8_kpi = DurationKpi(
    'NEXTVLAD_time_card8',
    0.08,
    0,
    actived=True,
    desc='train time in four GPU card')
STNET_loss_card1_kpi = CostKpi(
    'STNET_loss_card1', 0.08, 0, actived=True, desc='train cost')
STNET_time_card1_kpi = DurationKpi(
    'STNET_time_card1',
    0.08,
    0,
    actived=True,
    desc='train speed in one GPU card')
STNET_loss_card8_kpi = CostKpi(
    'STNET_loss_card8', 0.08, 0, actived=True, desc='train cost')
STNET_time_card8_kpi = DurationKpi(
    'STNET_time_card8',
    0.08,
    0,
    actived=True,
    desc='train time in four GPU card')
TSM_loss_card1_kpi = CostKpi(
    'TSM_loss_card1', 0.08, 0, actived=True, desc='train cost')
TSM_time_card1_kpi = DurationKpi(
    'TSM_time_card1',
    0.08,
    0,
    actived=True,
    desc='train speed in one GPU card')
TSM_loss_card8_kpi = CostKpi(
    'TSM_loss_card8', 0.08, 0, actived=True, desc='train cost')
TSM_time_card8_kpi = DurationKpi(
    'TSM_time_card8',
    0.08,
    0,
    actived=True,
    desc='train speed in one GPU card')
TSN_loss_card1_kpi = CostKpi(
    'TSN_loss_card1', 0.08, 0, actived=True, desc='train cost')
TSN_time_card1_kpi = DurationKpi(
    'TSN_time_card1',
    0.08,
    0,
    actived=True,
    desc='train speed in one GPU card')
TSN_loss_card8_kpi = CostKpi(
    'TSN_loss_card8', 0.08, 0, actived=True, desc='train cost')
TSN_time_card8_kpi = DurationKpi(
    'TSN_time_card8',
    0.08,
    0,
    actived=True,
    desc='train time in four GPU card')
NONLOCAL_loss_card1_kpi = CostKpi(
    'NONLOCAL_loss_card1', 0.08, 0, actived=True, desc='train cost')
NONLOCAL_time_card1_kpi = DurationKpi(
    'NONLOCAL_time_card1',
    0.08,
    0,
    actived=True,
    desc='train speed in one GPU card')
NONLOCAL_loss_card8_kpi = CostKpi(
    'NONLOCAL_loss_card8', 0.08, 0, actived=True, desc='train cost')
NONLOCAL_time_card8_kpi = DurationKpi(
    'NONLOCAL_time_card8',
    0.08,
    0,
    actived=True,
    desc='train time in four GPU card')

tracking_kpis = [
    AttentionCluster_loss_card1_kpi, AttentionCluster_time_card1_kpi,
    AttentionCluster_loss_card8_kpi, AttentionCluster_time_card8_kpi,
    AttentionLSTM_loss_card1_kpi, AttentionLSTM_time_card1_kpi,
    AttentionLSTM_loss_card8_kpi, AttentionLSTM_time_card8_kpi,
    NEXTVLAD_loss_card1_kpi, NEXTVLAD_time_card1_kpi,
    NEXTVLAD_loss_card8_kpi, NEXTVLAD_time_card8_kpi, 
    STNET_loss_card1_kpi, STNET_time_card1_kpi, STNET_loss_card8_kpi, 
    STNET_time_card8_kpi, TSM_loss_card1_kpi, TSM_time_card1_kpi, 
    TSM_loss_card8_kpi, TSM_time_card8_kpi, TSN_loss_card1_kpi, 
    TSN_time_card1_kpi, TSN_loss_card8_kpi, TSN_time_card8_kpi,
    NONLOCAL_loss_card1_kpi, NONLOCAL_time_card1_kpi, NONLOCAL_loss_card8_kpi,
    NONLOCAL_time_card8_kpi
]


def parse_log(log):
    '''
    This method should be implemented by model developers.
    The suggestion:
    each line in the log should be key, value, for example:
    "
    tsm_loss_card1\t1.0
    tsm_time_card1\t1.0
    tsm_loss_card8\t1.0
    tsm_time_card8\t1.0
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
