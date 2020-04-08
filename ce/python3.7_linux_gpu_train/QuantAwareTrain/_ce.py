# this file is only used for continuous evaluation test!

import os
import sys

sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi

QAT_GoogleNet_test_acc5_kpi = CostKpi('QAT_GoogleNet_test_acc5', 0.08, 0, actived=True, desc='train cost')
QAT_GoogleNet_test_acc1_kpi = CostKpi('QAT_GoogleNet_test_acc1', 0.08, 0, actived=True, desc='train cost')

QAT_MobileNet_test_acc5_kpi = CostKpi('QAT_MobileNet_test_acc5', 0.08, 0, actived=True, desc='train cost')
QAT_MobileNet_test_acc1_kpi = CostKpi('QAT_MobileNet_test_acc1', 0.08, 0, actived=True, desc='train cost')

QAT_MobileNetV2_test_acc5_kpi = CostKpi('QAT_MobileNetV2_test_acc5', 0.08, 0, actived=True, desc='train cost')
QAT_MobileNetV2_test_acc1_kpi = CostKpi('QAT_MobileNetV2_test_acc1', 0.08, 0, actived=True, desc='train cost')

QAT_ResNet50_test_acc5_kpi = CostKpi('QAT_ResNet50_test_acc5', 0.08, 0, actived=True, desc='train cost')
QAT_ResNet50_test_acc1_kpi = CostKpi('QAT_ResNet50_test_acc1', 0.08, 0, actived=True, desc='train cost')

QAT_VGG16_test_acc5_kpi = CostKpi('QAT_VGG16_test_acc5', 0.08, 0, actived=True, desc='train cost')
QAT_VGG16_test_acc1_kpi = CostKpi('QAT_VGG16_test_acc1', 0.08, 0, actived=True, desc='train cost')

tracking_kpis = [QAT_GoogleNet_test_acc5_kpi, QAT_GoogleNet_test_acc1_kpi,
                 QAT_MobileNet_test_acc5_kpi, QAT_MobileNet_test_acc1_kpi,
                 QAT_MobileNetV2_test_acc5_kpi, QAT_MobileNetV2_test_acc1_kpi,
                 QAT_ResNet50_test_acc5_kpi, QAT_ResNet50_test_acc1_kpi,
                 QAT_VGG16_test_acc5_kpi, QAT_VGG16_test_acc1_kpi]


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
    def _remove_comma(str_data):
        data = str_data.split(',')[0]
        return float(data)

    for line in log.split('\n'):
        fs = line.strip().split(' ')
        if len(fs) < 3 and fs[0] == 'model:':
            model_name = fs[1]
            print(model_name)

        if len(fs) > 3 and fs[0] == 'End':
            loss_id = fs.index("test_loss")
            acc1_id = fs.index("test_acc1")
            acc5_id = fs.index("test_acc5")
            yield {"QAT_{}_test_acc1".format(model_name): _remove_comma(fs[acc1_id+1]),
                   "QAT_{}_test_acc5".format(model_name): _remove_comma(fs[acc5_id+1])}


def log_to_ce(log):
    kpi_tracker = {}
    for kpi in tracking_kpis:
        kpi_tracker[kpi.name] = kpi

    for kpis in parse_log(log):
        print(kpis)
        for key in kpis:
            kpi_tracker[key].add_record(kpis[key])
            kpi_tracker[key].persist()


if __name__ == '__main__':
    log = sys.stdin.read()
    log_to_ce(log)
