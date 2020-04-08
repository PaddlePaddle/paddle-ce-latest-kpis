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
DPN107_loss_card1_kpi = CostKpi('DPN107_loss_card1', 0.05, 0, actived=True, desc='train cost')
DPN107_time_card1_kpi = DurationKpi(
    'DPN107_time_card1',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in one GPU card')
DPN107_loss_card8_kpi = CostKpi('DPN107_loss_card8', 0.02, 0, actived=True, desc='train cost')
DPN107_time_card8_kpi = DurationKpi(
    'DPN107_time_card8',
    0.05,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
DarkNet53_loss_card1_kpi = CostKpi('DarkNet53_loss_card1', 0.05, 0, actived=True, desc='train cost')
DarkNet53_time_card1_kpi = DurationKpi(
    'DarkNet53_time_card1',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in one GPU card')
DarkNet53_loss_card8_kpi = CostKpi('DarkNet53_loss_card8', 0.02, 0, actived=True, desc='train cost')
DarkNet53_time_card8_kpi = DurationKpi(
    'DarkNet53_time_card8',
    0.05,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
DenseNet121_loss_card1_kpi = CostKpi('DenseNet121_loss_card1', 0.05, 0, actived=True, desc='train cost')
DenseNet121_time_card1_kpi = DurationKpi(
    'DenseNet121_time_card1',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in one GPU card')
DenseNet121_loss_card8_kpi = CostKpi('DenseNet121_loss_card8', 0.02, 0, actived=True, desc='train cost')
DenseNet121_time_card8_kpi = DurationKpi(
    'DenseNet121_time_card8',
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
HRNet_W18_C_loss_card1_kpi = CostKpi('HRNet_W18_C_loss_card1', 0.02, 0, actived=True, desc='train cost')
HRNet_W18_C_time_card1_kpi = DurationKpi(
    'HRNet_W18_C_time_card1',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in one GPU card')
HRNet_W18_C_loss_card8_kpi = CostKpi('HRNet_W18_C_loss_card8', 0.02, 0, actived=True, desc='train cost')
HRNet_W18_C_time_card8_kpi = DurationKpi(
    'HRNet_W18_C_time_card8',
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
Res2Net50_vd_26w_4s_loss_card1_kpi = CostKpi('Res2Net50_vd_26w_4s_loss_card1', 0.02, 0, actived=True, desc='train cost')
Res2Net50_vd_26w_4s_time_card1_kpi = DurationKpi(
    'Res2Net50_vd_26w_4s_time_card1',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 1 GPU card')
Res2Net50_vd_26w_4s_loss_card8_kpi = CostKpi('Res2Net50_vd_26w_4s_loss_card8', 0.02, 0, actived=True, desc='train cost')
Res2Net50_vd_26w_4s_time_card8_kpi = DurationKpi(
    'Res2Net50_vd_26w_4s_time_card8',
    0.05,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
ResNeXt101_32x4d_loss_card1_kpi = CostKpi('ResNeXt101_32x4d_loss_card1', 0.02, 0, actived=True, desc='train cost')
ResNeXt101_32x4d_time_card1_kpi = DurationKpi(
    'ResNeXt101_32x4d_time_card1',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 1 GPU card')
ResNeXt101_32x4d_loss_card8_kpi = CostKpi('ResNeXt101_32x4d_loss_card8', 0.02, 0, actived=True, desc='train cost')
ResNeXt101_32x4d_time_card8_kpi = DurationKpi(
    'ResNeXt101_32x4d_time_card8',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
ResNet152_vd_loss_card1_kpi = CostKpi('ResNet152_vd_loss_card1', 0.02, 0, actived=True, desc='train cost')
ResNet152_vd_time_card1_kpi = DurationKpi(
    'ResNet152_vd_time_card1',
    0.05,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 1 GPU card')
ResNet152_vd_loss_card8_kpi = CostKpi('ResNet152_vd_loss_card8', 0.02, 0, actived=True, desc='train cost')
ResNet152_vd_time_card8_kpi = DurationKpi(
    'ResNet152_vd_time_card8',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
SE_ResNeXt50_vd_32x4d_loss_card1_kpi = CostKpi('SE_ResNeXt50_vd_32x4d_loss_card1', 0.05, 0, actived=True, desc='train cost')
SE_ResNeXt50_vd_32x4d_time_card1_kpi = DurationKpi(
    'SE_ResNeXt50_vd_32x4d_time_card1',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 1 GPU card')
SE_ResNeXt50_vd_32x4d_loss_card8_kpi = CostKpi('SE_ResNeXt50_vd_32x4d_loss_card8', 0.02, 0, actived=True, desc='train cost')
SE_ResNeXt50_vd_32x4d_time_card8_kpi = DurationKpi(
    'SE_ResNeXt50_vd_32x4d_time_card8',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
ShuffleNetV2_swish_loss_card1_kpi = CostKpi('ShuffleNetV2_swish_loss_card1', 0.02, 0, actived=True, desc='train cost')
ShuffleNetV2_swish_time_card1_kpi = DurationKpi(
    'ShuffleNetV2_swish_time_card1',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 1 GPU card')
ShuffleNetV2_swish_loss_card8_kpi = CostKpi('ShuffleNetV2_swish_loss_card8', 0.02, 0, actived=True, desc='train cost')
ShuffleNetV2_swish_time_card8_kpi = DurationKpi(
    'ShuffleNetV2_swish_time_card8',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
SqueezeNet1_1_loss_card1_kpi = CostKpi('SqueezeNet1_1_loss_card1', 0.02, 0, actived=True, desc='train cost')
SqueezeNet1_1_time_card1_kpi = DurationKpi(
    'SqueezeNet1_1_time_card1',
    0.1,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 1 GPU card')
SqueezeNet1_1_loss_card8_kpi = CostKpi('SqueezeNet1_1_loss_card8', 0.02, 0, actived=True, desc='train cost')
SqueezeNet1_1_time_card8_kpi = DurationKpi(
    'SqueezeNet1_1_time_card8',
    0.02,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
VGG19_loss_card1_kpi = CostKpi('VGG19_loss_card1', 0.02, 0, actived=True, desc='train cost')
VGG19_time_card1_kpi = DurationKpi(
    'VGG19_time_card1',
    0.05,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 1 GPU card')
VGG19_loss_card8_kpi = CostKpi('VGG19_loss_card8', 0.05, 0, actived=True, desc='train cost')
VGG19_time_card8_kpi = DurationKpi(
    'VGG19_time_card8',
    0.1,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
Xception65_deeplab_loss_card1_kpi = CostKpi('Xception65_deeplab_loss_card1', 0.02, 0, actived=True, desc='train cost')
Xception65_deeplab_time_card1_kpi = DurationKpi(
    'Xception65_deeplab_time_card1',
    0.1,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 1 GPU card')
Xception65_deeplab_loss_card8_kpi = CostKpi('Xception65_deeplab_loss_card8', 0.02, 0, actived=True, desc='train cost')
Xception65_deeplab_time_card8_kpi = DurationKpi(
    'Xception65_deeplab_time_card8',
    0.05,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train time in 8 GPU card')
tracking_kpis = [ AlexNet_loss_card1_kpi, AlexNet_time_card1_kpi, AlexNet_loss_card8_kpi, AlexNet_time_card8_kpi, DPN107_loss_card1_kpi, DPN107_time_card1_kpi, DPN107_loss_card8_kpi, DPN107_time_card8_kpi, DarkNet53_loss_card1_kpi, DarkNet53_time_card1_kpi, DarkNet53_loss_card8_kpi, DarkNet53_time_card8_kpi, DenseNet121_loss_card1_kpi, DenseNet121_time_card1_kpi, DenseNet121_loss_card8_kpi, DenseNet121_time_card8_kpi, EfficientNet_loss_card1_kpi, EfficientNet_time_card1_kpi, EfficientNet_loss_card8_kpi, EfficientNet_time_card8_kpi, GoogLeNet_loss_card1_kpi, GoogLeNet_time_card1_kpi, GoogLeNet_loss_card8_kpi, GoogLeNet_time_card8_kpi, HRNet_W18_C_loss_card1_kpi, HRNet_W18_C_time_card1_kpi, HRNet_W18_C_loss_card8_kpi, HRNet_W18_C_time_card8_kpi, InceptionV4_loss_card1_kpi, InceptionV4_time_card1_kpi, InceptionV4_loss_card8_kpi, InceptionV4_time_card8_kpi, MobileNetV1_loss_card1_kpi, MobileNetV1_time_card1_kpi, MobileNetV1_loss_card8_kpi, MobileNetV1_time_card8_kpi, MobileNetV2_loss_card1_kpi, MobileNetV2_time_card1_kpi, MobileNetV2_loss_card8_kpi, MobileNetV2_time_card8_kpi, Res2Net50_vd_26w_4s_loss_card1_kpi, Res2Net50_vd_26w_4s_time_card1_kpi, Res2Net50_vd_26w_4s_loss_card8_kpi,Res2Net50_vd_26w_4s_time_card8_kpi, ResNeXt101_32x4d_loss_card1_kpi, ResNeXt101_32x4d_time_card1_kpi, ResNeXt101_32x4d_loss_card8_kpi, ResNeXt101_32x4d_time_card8_kpi, ResNet152_vd_loss_card1_kpi, ResNet152_vd_time_card1_kpi, ResNet152_vd_loss_card8_kpi, ResNet152_vd_time_card8_kpi, SE_ResNeXt50_vd_32x4d_loss_card1_kpi, SE_ResNeXt50_vd_32x4d_time_card1_kpi, SE_ResNeXt50_vd_32x4d_loss_card8_kpi, SE_ResNeXt50_vd_32x4d_time_card8_kpi, ShuffleNetV2_swish_loss_card1_kpi, ShuffleNetV2_swish_time_card1_kpi, ShuffleNetV2_swish_loss_card8_kpi, ShuffleNetV2_swish_time_card8_kpi, SqueezeNet1_1_loss_card1_kpi, SqueezeNet1_1_time_card1_kpi, SqueezeNet1_1_loss_card8_kpi, SqueezeNet1_1_time_card8_kpi, VGG19_loss_card1_kpi, VGG19_time_card1_kpi, VGG19_loss_card8_kpi, VGG19_time_card8_kpi, Xception65_deeplab_loss_card1_kpi, Xception65_deeplab_time_card1_kpi, Xception65_deeplab_loss_card8_kpi, Xception65_deeplab_time_card8_kpi]

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
