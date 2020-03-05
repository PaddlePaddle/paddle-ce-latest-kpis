# this file is only used for continuous evaluation test!

import os
import sys

sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, AccKpi
 
deeplabv3p_loss_card1_kpi = CostKpi('deeplabv3p_loss_card1', 0.1, 0, actived=True, 
                                    desc='train loss in 1 GPU card')
deeplabv3p_speed_card1_kpi = AccKpi('deeplabv3p_speed_card1', 0.05, 0, actived=True,
                                         desc='train speed in 1 GPU card')
deeplabv3p_loss_card8_kpi = CostKpi('deeplabv3p_loss_card8', 0.03, 0, actived=True, 
                                    desc='train loss in 8 GPU card')
deeplabv3p_speed_card8_kpi = AccKpi('deeplabv3p_speed_card8', 0.03, 0, actived=True,
                                        desc='train speed in 8 GPU card')
icnet_loss_card1_kpi = CostKpi('icnet_loss_card1', 0.03, 0, actived=True,
                               desc='train loss in 1 GPU card')
icnet_speed_card1_kpi = AccKpi('icnet_speed_card1', 0.08, 0, actived=True,
                                   desc='train speed in 1 GPU card')
icnet_loss_card8_kpi = CostKpi('icnet_loss_card8', 0.03, 0, actived=True,
                               desc='train loss in 8 GPU card')
icnet_speed_card8_kpi = AccKpi('icnet_speed_card8', 0.03, 0, actived=True,
                                   desc='train speed in 8 GPU card')
unet_loss_card1_kpi = CostKpi('unet_loss_card1', 0.03, 0, actived=True,
                              desc='train loss in 1 GPU card')
unet_speed_card1_kpi = AccKpi('unet_speed_card1', 0.03, 0, actived=True,
                                  desc='train speed in 1 GPU card')
unet_loss_card8_kpi = CostKpi('unet_loss_card8', 0.03, 0, actived=True,
                              desc='train loss in 8 GPU card')
unet_speed_card8_kpi = AccKpi('unet_speed_card8', 0.03, 0, actived=True,
                                  desc='train speed in 8 GPU card')
pspnet_loss_card1_kpi = CostKpi('pspnet_loss_card1', 0.08, 0, actived=True,
                                desc='train loss in 1 GPU card')
pspnet_speed_card1_kpi = AccKpi('pspnet_speed_card1', 0.03, 0, actived=True,
                                    desc='train speed in 1 GPU card')
pspnet_loss_card8_kpi = CostKpi('pspnet_loss_card8', 0.03, 0, actived=True,
                                desc='train loss in 8 GPU card')
pspnet_speed_card8_kpi = AccKpi('pspnet_speed_card8', 0.03, 0, actived=True,
                                    desc='train speed in 8 GPU card')
hrnet_loss_card1_kpi = CostKpi('hrnet_loss_card1', 0.08, 0, actived=True,
                               desc='train loss in 1 GPU card')
hrnet_speed_card1_kpi = AccKpi('hrnet_speed_card1', 0.08, 0, actived=True,
                                   desc='train speed in 1 GPU card')
hrnet_loss_card8_kpi = CostKpi('hrnet_loss_card8', 0.08, 0, actived=True,
                               desc='train loss in 8 GPU card')
hrnet_speed_card8_kpi = AccKpi('hrnet_speed_card8', 0.08, 0, actived=True,
                                   desc='train speed in 8 GPU card')
fastscnn_loss_card1_kpi = CostKpi('fastscnn_loss_card1', 0.08, 0, actived=True,
                               desc='train loss in 1 GPU card')
fastscnn_speed_card1_kpi = AccKpi('fastscnn_speed_card1', 0.08, 0, actived=True,
                                   desc='train speed in 1 GPU card')
fastscnn_loss_card8_kpi = CostKpi('fastscnn_loss_card8', 0.08, 0, actived=True,
                               desc='train loss in 8 GPU card')
fastscnn_speed_card8_kpi = AccKpi('fastscnn_speed_card8', 0.08, 0, actived=True,
                                   desc='train speed in 8 GPU card')
tracking_kpis = [deeplabv3p_loss_card1_kpi, deeplabv3p_speed_card1_kpi, deeplabv3p_loss_card8_kpi,
                 deeplabv3p_speed_card8_kpi, icnet_loss_card1_kpi, icnet_speed_card1_kpi, icnet_loss_card8_kpi,
                 icnet_speed_card8_kpi, unet_loss_card1_kpi, unet_speed_card1_kpi, unet_loss_card8_kpi,
                 unet_speed_card8_kpi, pspnet_loss_card1_kpi, pspnet_speed_card1_kpi, pspnet_loss_card8_kpi,
                 pspnet_speed_card8_kpi, hrnet_loss_card1_kpi, hrnet_speed_card1_kpi, hrnet_loss_card8_kpi,
                 hrnet_speed_card8_kpi, fastscnn_loss_card1_kpi, fastscnn_speed_card1_kpi, fastscnn_loss_card8_kpi, fastscnn_speed_card8_kpi]


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
