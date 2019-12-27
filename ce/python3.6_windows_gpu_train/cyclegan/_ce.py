####this file is only used for continuous evaluation test!
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

#### NOTE kpi.py should shared in models in some way!!!!

cyclegan_g_A_loss_card1_kpi = CostKpi(
    'cyclegan_g_A_loss_card1', 0.1, 0, actived=True, desc='g_A_loss')
cyclegan_g_A_cyc_loss_card1_kpi = CostKpi(
    'cyclegan_g_A_cyc_loss_card1', 0.02, 0, actived=True, desc='g_A_cyc_loss')
cyclegan_g_A_idt_loss_card1_kpi = CostKpi(
    'cyclegan_g_A_idt_loss_card1', 0.02, 0, actived=True, desc='g_A_idt_loss')
cyclegan_d_A_loss_card1_kpi = CostKpi(
    'cyclegan_d_A_loss_card1', 0.05, 0, actived=True, desc='d_A_loss')
cyclegan_g_B_loss_card1_kpi = CostKpi(
    'cyclegan_g_B_loss_card1', 0.1, 0, actived=True, desc='g_B_loss')
cyclegan_g_B_cyc_loss_card1_kpi = CostKpi(
    'cyclegan_g_B_cyc_loss_card1', 0.02, 0, actived=True, desc='g_B_cyc_loss')
cyclegan_g_B_idt_loss_card1_kpi = CostKpi(
    'cyclegan_g_B_idt_loss_card1', 0.02, 0, actived=True, desc='g_B_idt_loss')
cyclegan_d_B_loss_card1_kpi = CostKpi(
    'cyclegan_d_B_loss_card1', 0.08, 0, actived=True, desc='d_B_loss')
cyclegan_Batch_time_cost_card1_kpi = DurationKpi(
    'cyclegan_Batch_time_cost_card1', 0.05, 0, actived=True, desc='Batch_time_cost')

tracking_kpis = [
     cyclegan_g_A_loss_card1_kpi, cyclegan_g_A_cyc_loss_card1_kpi, cyclegan_g_A_idt_loss_card1_kpi,
     cyclegan_d_A_loss_card1_kpi, cyclegan_g_B_loss_card1_kpi, cyclegan_g_B_cyc_loss_card1_kpi,
     cyclegan_g_B_idt_loss_card1_kpi, cyclegan_d_B_loss_card1_kpi, cyclegan_Batch_time_cost_card1_kpi, 
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
