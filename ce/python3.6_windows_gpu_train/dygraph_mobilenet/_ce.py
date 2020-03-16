#This file is only used for continuous evaluation test!
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

#### NOTE kpi.py should shared in models in some way!!!!

MobileNetV1_train_loss_card1_kpi = CostKpi('MobileNetV1_train_loss_card1', 0.08, 0, actived=True, desc='train cost')
MobileNetV1_eval_loss_card1_kpi = CostKpi('MobileNetV1_eval_loss_card1', 0.08, 0, actived=True, desc='eval cost')

MobileNetV2_train_loss_card1_kpi = CostKpi('MobileNetV2_train_loss_card1', 0.08, 0, actived=True, desc='train cost')
MobileNetV2_eval_loss_card1_kpi = CostKpi('MobileNetV2_eval_loss_card1', 0.08, 0, actived=True, desc='eval cost')

tracking_kpis = [ MobileNetV1_train_loss_card1_kpi, MobileNetV1_eval_loss_card1_kpi, MobileNetV2_train_loss_card1_kpi, MobileNetV2_eval_loss_card1_kpi] 

def parse_log(log):
    '''
    This method should be implemented by model developers.
    The suggestion:
    each line in the log should be key, value, for example:
    "
    
    "
    '''
    for line in log.split('\n'):
        fs = line.strip().split('\t')
        print(fs)
        if len(fs) == 3 and fs[0] == 'kpis':
            print("-----%s" % fs)
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
    print("*****")
    print(log)
    print("****")
    log_to_ce(log)
