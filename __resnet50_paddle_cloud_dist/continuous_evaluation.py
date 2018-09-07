import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import AccKpi

flowers_resnet50_dist_train_acc_kpi = AccKpi('flowers_resnet50_dist_train_acc', 0.08, 0, actived=True)
flowers_resnet50_dist_train_speed_kpi = AccKpi('flowers_resnet50_dist_train_speed', 0.1, 0, actived=True)

tracking_kpis = [
    flowers_resnet50_dist_train_speed_kpi,
    flowers_resnet50_dist_train_acc_kpi,
]
