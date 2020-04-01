# this file is only used for continuous evaluation test!

import os
import sys

sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi

cascade_rcnn_r50_fpn_1x_loss_kpi = CostKpi('cascade_rcnn_r50_fpn_1x_loss', 0.08, 0, actived=True, desc='train cost')
cascade_rcnn_r50_fpn_1x_time_kpi = DurationKpi('cascade_rcnn_r50_fpn_1x_time', 0.08, 0, actived=True,
                                         desc='train speed in 8 GPU card')
faster_rcnn_r50_fpn_1x_loss_kpi = CostKpi('faster_rcnn_r50_fpn_1x_loss', 0.3, 0, actived=True, desc='train cost')
faster_rcnn_r50_fpn_1x_time_kpi = DurationKpi('faster_rcnn_r50_fpn_1x_time', 0.08, 0, actived=True,
                                        desc='train speed in 8 GPU card')
mask_rcnn_r50_fpn_1x_loss_kpi = CostKpi('mask_rcnn_r50_fpn_1x_loss', 0.08, 0, actived=True, desc='train cost')
mask_rcnn_r50_fpn_1x_time_kpi = DurationKpi('mask_rcnn_r50_fpn_1x_time', 0.08, 0, actived=True, desc='train speed in 8 GPU card')
mask_rcnn_r101_vd_fpn_1x_loss_kpi = CostKpi('mask_rcnn_r101_vd_fpn_1x_loss', 0.08, 0, actived=True, desc='train cost')
mask_rcnn_r101_vd_fpn_1x_time_kpi = DurationKpi('mask_rcnn_r101_vd_fpn_1x_time', 0.08, 0, actived=True, desc='train speed in 8 GPU card')
retinanet_r50_fpn_1x_loss_kpi = CostKpi('retinanet_r50_fpn_1x_loss', 0.08, 0, actived=True, desc='train cost')
retinanet_r50_fpn_1x_time_kpi = DurationKpi('retinanet_r50_fpn_1x_time', 0.08, 0, actived=True, desc='train speed in 8 GPU card')
yolov3_r50vd_dcn_obj365_pretrained_coco_loss_kpi = CostKpi('yolov3_r50vd_dcn_obj365_pretrained_coco_loss', 0.08, 0, actived=True, desc='train cost')
yolov3_r50vd_dcn_obj365_pretrained_coco_time_kpi = DurationKpi('yolov3_r50vd_dcn_obj365_pretrained_coco_time', 0.08, 0, actived=True, desc='train speed in 8 GPU card')
yolov3_darknet_loss_kpi = CostKpi('yolov3_darknet_loss', 0.08, 0, actived=True, desc='train cost')
yolov3_darknet_time_kpi = DurationKpi('yolov3_darknet_time', 0.08, 0, actived=True, desc='train speed in 8 GPU card')
yolov3_r34_voc_loss_kpi = CostKpi('yolov3_r34_voc_loss', 0.08, 0, actived=True, desc='train cost')
yolov3_r34_voc_time_kpi = DurationKpi('yolov3_r34_voc_time', 0.08, 0, actived=True, desc='train speed in 8 GPU card')
blazeface_nas_loss_kpi = CostKpi('blazeface_nas_loss', 0.08, 0, actived=True, desc='train cost')
blazeface_nas_time_kpi = DurationKpi('blazeface_nas_time', 0.08, 0, actived=True, desc='train speed in 8 GPU card')
tracking_kpis = [cascade_rcnn_r50_fpn_1x_loss_kpi, cascade_rcnn_r50_fpn_1x_time_kpi, 
                 faster_rcnn_r50_fpn_1x_loss_kpi, faster_rcnn_r50_fpn_1x_time_kpi,
                 mask_rcnn_r50_fpn_1x_loss_kpi, mask_rcnn_r50_fpn_1x_time_kpi,
                 mask_rcnn_r101_vd_fpn_1x_loss_kpi, mask_rcnn_r101_vd_fpn_1x_time_kpi,
                 retinanet_r50_fpn_1x_loss_kpi, retinanet_r50_fpn_1x_time_kpi,
                 yolov3_r50vd_dcn_obj365_pretrained_coco_loss_kpi, yolov3_r50vd_dcn_obj365_pretrained_coco_time_kpi,
                 yolov3_darknet_loss_kpi, yolov3_darknet_time_kpi,
                 yolov3_r34_voc_loss_kpi, yolov3_r34_voc_time_kpi,
                 blazeface_nas_loss_kpi, blazeface_nas_time_kpi]


def parse_log(log):
    '''
    This method should be implemented by model developers.
    '''
    fs = log.strip().split(' ')
    print(fs)
    loss_kpi_name = fs[-12]
    loss_kpi_value = float(fs[-9])
    time_kpi_name = fs[-6]
    time_kpi_value = float(fs[-4])
    yield loss_kpi_name, loss_kpi_value
    yield time_kpi_name, time_kpi_value


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
