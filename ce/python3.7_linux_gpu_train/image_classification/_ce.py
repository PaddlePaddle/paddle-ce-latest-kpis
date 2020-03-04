####this file is only used for continuous evaluation test!
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

#### NOTE kpi.py should shared in models in some way!!!!

AlexNet_loss_card1_kpi = CostKpi('AlexNet_loss_card1', 0.02, 0, actived=True, desc='train cost')
AlexNet_time_card1_kpi = DurationKpi(
    'AlexNet_time_card1',
    0.05,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in one GPU card')
AlexNet_loss_card8_kpi = CostKpi('AlexNet_loss_card8', 0.02, 0, actived=True, desc='train cost')
AlexNet_time_card8_kpi = DurationKpi(
    'AlexNet_time_card8',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
DPN_loss_card1_kpi = CostKpi('DPN_loss_card1', 0.05, 0, actived=True, desc='train cost')
DPN_time_card1_kpi = DurationKpi(
    'DPN_time_card1',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in one GPU card')
DPN_loss_card8_kpi = CostKpi('DPN_loss_card8', 0.02, 0, actived=True, desc='train cost')
DPN_time_card8_kpi = DurationKpi(
    'DPN_time_card8',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
DarkNet_loss_card1_kpi = CostKpi('DarkNet_loss_card1', 0.05, 0, actived=True, desc='train cost')
DarkNet_time_card1_kpi = DurationKpi(
    'DarkNet_time_card1',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in one GPU card')
DarkNet_loss_card8_kpi = CostKpi('DarkNet_loss_card8', 0.02, 0, actived=True, desc='train cost')
DarkNet_time_card8_kpi = DurationKpi(
    'DarkNet_time_card8',
    0.05,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
DenseNet_loss_card1_kpi = CostKpi('DenseNet_loss_card1', 0.05, 0, actived=True, desc='train cost')
DenseNet_time_card1_kpi = DurationKpi(
    'DenseNet_time_card1',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in one GPU card')
DenseNet_loss_card8_kpi = CostKpi('DenseNet_loss_card8', 0.02, 0, actived=True, desc='train cost')
DenseNet_time_card8_kpi = DurationKpi(
    'DenseNet_time_card8',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
EfficientNet_loss_card1_kpi = CostKpi('EfficientNet_loss_card1', 0.02, 0, actived=True, desc='train cost')
EfficientNet_time_card1_kpi = DurationKpi(
    'EfficientNet_time_card1',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in one GPU card')
EfficientNet_loss_card8_kpi = CostKpi('EfficientNet_loss_card8', 0.02, 0, actived=True, desc='train cost')
EfficientNet_time_card8_kpi = DurationKpi(
    'EfficientNet_time_card8',
    0.1,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
GoogLeNet_loss_card1_kpi = CostKpi('GoogLeNet_loss_card1', 0.02, 0, actived=True, desc='train cost')
GoogLeNet_time_card1_kpi = DurationKpi(
    'GoogLeNet_time_card1',
    0.1,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in one GPU card')
GoogLeNet_loss_card8_kpi = CostKpi('GoogLeNet_loss_card8', 0.02, 0, actived=True, desc='train cost')
GoogLeNet_time_card8_kpi = DurationKpi(
    'GoogLeNet_time_card8',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
HRNet_loss_card1_kpi = CostKpi('HRNet_loss_card1', 0.02, 0, actived=True, desc='train cost')
HRNet_time_card1_kpi = DurationKpi(
    'HRNet_time_card1',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in one GPU card')
HRNet_loss_card8_kpi = CostKpi('HRNet_loss_card8', 0.02, 0, actived=True, desc='train cost')
HRNet_time_card8_kpi = DurationKpi(
    'HRNet_time_card8',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
InceptionV4_loss_card1_kpi = CostKpi('InceptionV4_loss_card1', 0.02, 0, actived=True, desc='train cost')
InceptionV4_time_card1_kpi = DurationKpi(
    'InceptionV4_time_card1',
    0.05,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in one GPU card')
InceptionV4_loss_card8_kpi = CostKpi('InceptionV4_loss_card8', 0.02, 0, actived=True, desc='train cost')
InceptionV4_time_card8_kpi = DurationKpi(
    'InceptionV4_time_card8',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
MobileNetV1_loss_card1_kpi = CostKpi('MobileNetV1_loss_card1', 0.05, 0, actived=True, desc='train cost')
MobileNetV1_time_card1_kpi = DurationKpi(
    'MobileNetV1_time_card1',
    0.1,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in one GPU card')
MobileNetV1_loss_card8_kpi = CostKpi('MobileNetV1_loss_card8', 0.02, 0, actived=True, desc='train cost')
MobileNetV1_time_card8_kpi = DurationKpi(
    'MobileNetV1_time_card8',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
MobileNetV2_loss_card1_kpi = CostKpi('MobileNetV2_loss_card1', 0.02, 0, actived=True, desc='train cost')
MobileNetV2_time_card1_kpi = DurationKpi(
    'MobileNetV2_time_card1',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in one GPU card')
MobileNetV2_loss_card8_kpi = CostKpi('MobileNetV2_loss_card8', 0.02, 0, actived=True, desc='train cost')
MobileNetV2_time_card8_kpi = DurationKpi(
    'MobileNetV2_time_card8',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
Res2Net_loss_card1_kpi = CostKpi('Res2Net_loss_card1', 0.02, 0, actived=True, desc='train cost')
Res2Net_time_card1_kpi = DurationKpi(
    'Res2Net_time_card1',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 1 GPU card')
Res2Net_loss_card8_kpi = CostKpi('Res2Net_loss_card8', 0.02, 0, actived=True, desc='train cost')
Res2Net_time_card8_kpi = DurationKpi(
    'Res2Net_time_card8',
    0.05,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
ResNeXt_loss_card1_kpi = CostKpi('ResNeXt_loss_card1', 0.02, 0, actived=True, desc='train cost')
ResNeXt_time_card1_kpi = DurationKpi(
    'ResNeXt_time_card1',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 1 GPU card')
ResNeXt_loss_card8_kpi = CostKpi('ResNeXt_loss_card8', 0.02, 0, actived=True, desc='train cost')
ResNeXt_time_card8_kpi = DurationKpi(
    'ResNeXt_time_card8',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
ResNet_loss_card1_kpi = CostKpi('ResNet_loss_card1', 0.02, 0, actived=True, desc='train cost')
ResNet_time_card1_kpi = DurationKpi(
    'ResNet_time_card1',
    0.05,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 1 GPU card')
ResNet_loss_card8_kpi = CostKpi('ResNet_loss_card8', 0.02, 0, actived=True, desc='train cost')
ResNet_time_card8_kpi = DurationKpi(
    'ResNet_time_card8',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
SE_ResNeXt_loss_card1_kpi = CostKpi('SE_ResNeXt_loss_card1', 0.05, 0, actived=True, desc='train cost')
SE_ResNeXt_time_card1_kpi = DurationKpi(
    'SE_ResNeXt_time_card1',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 1 GPU card')
SE_ResNeXt_loss_card8_kpi = CostKpi('SE_ResNeXt_loss_card8', 0.02, 0, actived=True, desc='train cost')
SE_ResNeXt_time_card8_kpi = DurationKpi(
    'SE_ResNeXt_time_card8',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
ShuffleNet_loss_card1_kpi = CostKpi('ShuffleNet_loss_card1', 0.02, 0, actived=True, desc='train cost')
ShuffleNet_time_card1_kpi = DurationKpi(
    'ShuffleNet_time_card1',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 1 GPU card')
ShuffleNet_loss_card8_kpi = CostKpi('ShuffleNet_loss_card8', 0.02, 0, actived=True, desc='train cost')
ShuffleNet_time_card8_kpi = DurationKpi(
    'ShuffleNet_time_card8',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
SqueezeNet_loss_card1_kpi = CostKpi('SqueezeNet_loss_card1', 0.02, 0, actived=True, desc='train cost')
SqueezeNet_time_card1_kpi = DurationKpi(
    'SqueezeNet_time_card1',
    0.1,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 1 GPU card')
SqueezeNet_loss_card8_kpi = CostKpi('SqueezeNet_loss_card8', 0.02, 0, actived=True, desc='train cost')
SqueezeNet_time_card8_kpi = DurationKpi(
    'SqueezeNet_time_card8',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
VGG_loss_card1_kpi = CostKpi('VGG_loss_card1', 0.02, 0, actived=True, desc='train cost')
VGG_time_card1_kpi = DurationKpi(
    'VGG_time_card1',
    0.05,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 1 GPU card')
VGG_loss_card8_kpi = CostKpi('VGG_loss_card8', 0.05, 0, actived=True, desc='train cost')
VGG_time_card8_kpi = DurationKpi(
    'VGG_time_card8',
    0.1,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
Xception_loss_card1_kpi = CostKpi('Xception_loss_card1', 0.02, 0, actived=True, desc='train cost')
Xception_time_card1_kpi = DurationKpi(
    'Xception_time_card1',
    0.1,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 1 GPU card')
Xception_loss_card8_kpi = CostKpi('Xception_loss_card8', 0.02, 0, actived=True, desc='train cost')
Xception_time_card8_kpi = DurationKpi(
    'Xception_time_card8',
    0.05,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
tracking_kpis = [ AlexNet_loss_card1_kpi, AlexNet_time_card1_kpi, AlexNet_loss_card8_kpi, AlexNet_time_card8_kpi, DPN_loss_card1_kpi, DPN_time_card1_kpi, DPN_loss_card8_kpi, DPN_time_card8_kpi, DarkNet_loss_card1_kpi, DarkNet_time_card1_kpi, DarkNet_loss_card8_kpi, DarkNet_time_card8_kpi, DenseNet_loss_card1_kpi, DenseNet_time_card1_kpi, DenseNet_loss_card8_kpi, DenseNet_time_card8_kpi, EfficientNet_loss_card1_kpi, EfficientNet_time_card1_kpi, EfficientNet_loss_card8_kpi, EfficientNet_time_card8_kpi, GoogLeNet_loss_card1_kpi, GoogLeNet_time_card1_kpi, GoogLeNet_loss_card8_kpi, GoogLeNet_time_card8_kpi, HRNet_loss_card1_kpi, HRNet_time_card1_kpi, HRNet_loss_card8_kpi, HRNet_time_card8_kpi, InceptionV4_loss_card1_kpi, InceptionV4_time_card1_kpi, InceptionV4_loss_card8_kpi, InceptionV4_time_card8_kpi, MobileNetV1_loss_card1_kpi, MobileNetV1_time_card1_kpi, MobileNetV1_loss_card8_kpi, MobileNetV1_time_card8_kpi, MobileNetV2_loss_card1_kpi, MobileNetV2_time_card1_kpi, MobileNetV2_loss_card8_kpi, MobileNetV2_time_card8_kpi, Res2Net_loss_card1_kpi, Res2Net_time_card1_kpi, Res2Net_loss_card8_kpi,Res2Net_time_card8_kpi, ResNeXt_loss_card1_kpi, ResNeXt_time_card1_kpi, ResNeXt_loss_card8_kpi, ResNeXt_time_card8_kpi, ResNet_loss_card1_kpi, ResNet_time_card1_kpi, ResNet_loss_card8_kpi, ResNet_time_card8_kpi, SE_ResNeXt_loss_card1_kpi, SE_ResNeXt_time_card1_kpi, SE_ResNeXt_loss_card8_kpi, SE_ResNeXt_time_card8_kpi, ShuffleNet_loss_card1_kpi, ShuffleNet_time_card1_kpi, ShuffleNet_loss_card8_kpi, ShuffleNet_time_card8_kpi, SqueezeNet_loss_card1_kpi, SqueezeNet_time_card1_kpi, SqueezeNet_loss_card8_kpi, SqueezeNet_time_card8_kpi, VGG_loss_card1_kpi, VGG_time_card1_kpi, VGG_loss_card8_kpi, VGG_time_card8_kpi, Xception_loss_card1_kpi, Xception_time_card1_kpi, Xception_loss_card8_kpi, Xception_time_card8_kpi]

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
