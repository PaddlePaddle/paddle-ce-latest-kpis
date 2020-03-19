#This file is only used for continuous evaluation test!
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

#### NOTE kpi.py should shared in models in some way!!!!

train_loss_kpi = CostKpi('train_loss', 0.05, 0, actived=True, desc='train cost')
train_acc1_kpi = AccKpi('train_acc1', 0.05, 0, actived=True, desc='train acc1')
train_acc5_kpi = AccKpi('train_acc5', 0.05, 0, actived=True, desc='train acc5')
test_loss_kpi = CostKpi('test_loss', 0.05, 0, actived=True, desc='test cost')
test_acc1_kpi = AccKpi('test_acc1', 0.05, 0, actived=True, desc='test acc1')
test_acc5_kpi = AccKpi('test_acc5', 0.05, 0, actived=True, desc='test acc5')


tracking_kpis = [train_loss_kpi, train_acc1_kpi, train_acc5_kpi, test_loss_kpi, test_acc1_kpi, test_acc5_kpi]

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
