import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

cifar10_train_acc_kpi = AccKpi('cifar10_train_acc', 0.02, 0, actived=True)
cifar10_train_speed_kpi = AccKpi('cifar10_train_speed', 0.02, 0, actived=True)
flowers_train_speed_kpi = AccKpi('flowers_train_speed', 0.02, 0, actived=True)
cifar10_gpu_memory_kpi = DurationKpi('cifar10_gpu_memory', 0.02, 0)
flowers_gpu_memory_kpi = DurationKpi('flowers_gpu_memory', 0.02, 0)

tracking_kpis = [
    cifar10_train_acc_kpi,
    cifar10_train_speed_kpi,
    flowers_train_speed_kpi,
    cifar10_gpu_memory_kpi,
    flowers_gpu_memory_kpi,
]
