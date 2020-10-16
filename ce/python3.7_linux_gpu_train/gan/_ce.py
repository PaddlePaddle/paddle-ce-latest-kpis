# this file is only used for continuous evaluation test!

import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

# NOTE kpi.py should shared in models in some way!!!!

CGAN_dloss_kpi = CostKpi('CGAN_dloss', 0.08, 0, actived=True)
CGAN_gloss_kpi = CostKpi('CGAN_gloss', 0.08, 0, actived=True)
CGAN_time_kpi = DurationKpi('CGAN_time', 0.08, 0, actived=True)
DCGAN_dloss_kpi = CostKpi('DCGAN_dloss', 0.08, 0, actived=True)
DCGAN_gloss_kpi = CostKpi('DCGAN_gloss', 0.08, 0, actived=True)
DCGAN_time_kpi = DurationKpi('DCGAN_time', 0.08, 0, actived=True)
CycleGAN_dloss_kpi = CostKpi('CycleGAN_dloss', 0.08, 0, actived=True)
CycleGAN_gloss_kpi = CostKpi('CycleGAN_gloss', 0.08, 0, actived=True)
CycleGAN_time_kpi = DurationKpi('CycleGAN_time', 0.08, 0, actived=True)
SPADE_dloss_kpi = CostKpi('SPADE_dloss', 0.08, 0, actived=True)
SPADE_gloss_kpi = CostKpi('SPADE_gloss', 0.08, 0, actived=True)
SPADE_time_kpi = DurationKpi('SPADE_time', 0.08, 0, actived=True)
AttGAN_dloss_kpi = CostKpi('AttGAN_dloss', 0.08, 0, actived=True)
AttGAN_gloss_kpi = CostKpi('AttGAN_gloss', 0.08, 0, actived=True)
AttGAN_time_kpi = DurationKpi('AttGAN_time', 0.08, 0, actived=True)
StarGAN_dloss_kpi = CostKpi('StarGAN_dloss', 0.08, 0, actived=True)
StarGAN_gloss_kpi = CostKpi('StarGAN_gloss', 0.08, 0, actived=True)
StarGAN_time_kpi = DurationKpi('StarGAN_time', 0.08, 0, actived=True)
STGAN_dloss_kpi = CostKpi('STGAN_dloss', 0.08, 0, actived=True)
STGAN_gloss_kpi = CostKpi('STGAN_gloss', 0.08, 0, actived=True)
STGAN_time_kpi = DurationKpi('STGAN_time', 0.08, 0, actived=True)
Pix2pix_dloss_kpi = CostKpi('Pix2pix_dloss', 0.08, 0, actived=True)
Pix2pix_gloss_kpi = CostKpi('Pix2pix_gloss', 0.08, 0, actived=True)
Pix2pix_time_kpi = DurationKpi('Pix2pix_time', 0.08, 0, actived=True)

tracking_kpis = [CGAN_dloss_kpi, CGAN_gloss_kpi, CGAN_time_kpi,
                 DCGAN_dloss_kpi, DCGAN_gloss_kpi, DCGAN_time_kpi,
                 CycleGAN_dloss_kpi, CycleGAN_gloss_kpi, CycleGAN_time_kpi,
                 SPADE_dloss_kpi, SPADE_gloss_kpi, SPADE_time_kpi,
                 AttGAN_dloss_kpi, AttGAN_gloss_kpi, AttGAN_time_kpi,
                 StarGAN_dloss_kpi, StarGAN_gloss_kpi, StarGAN_time_kpi,
                 STGAN_dloss_kpi, STGAN_gloss_kpi, STGAN_time_kpi,
                 Pix2pix_dloss_kpi, Pix2pix_gloss_kpi, Pix2pix_time_kpi]


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
        if kpi_name in kpi_tracker:
            print(kpi_name, kpi_value)
            kpi_tracker[kpi_name].add_record(kpi_value)
            kpi_tracker[kpi_name].persist()


if __name__ == '__main__':
    log = sys.stdin.read()
    log_to_ce(log)
