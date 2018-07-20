import os
import sys

sys.path.append(os.environ['ceroot'])

from kpi import AccKpi
from kpi import DurationKpi

flowers_64_AllReduce_GPU_4_Cards_train_acc_kpi = AccKpi(
    'flowers_64_AllReduce_GPU_4_Cards_train_acc', 0.03, 0, actived=True)
flowers_64_AllReduce_GPU_4_Cards_train_speed_kpi = AccKpi(
    'flowers_64_AllReduce_GPU_4_Cards_train_speed', 0.06, 0, actived=True)
flowers_64_AllReduce_4_Cards_gpu_memory_kpi = DurationKpi(
    'flowers_64_AllReduce_4_Cards_gpu_memory', 0.1, 0, actived=True)

flowers_64_Reduce_GPU_4_Cards_train_acc_kpi = AccKpi(
    'flowers_64_Reduce_GPU_4_Cards_train_acc', 0.03, 0, actived=True)
flowers_64_Reduce_GPU_4_Cards_train_speed_kpi = AccKpi(
    'flowers_64_Reduce_GPU_4_Cards_train_speed', 0.06, 0, actived=True)
flowers_64_Reduce_4_Cards_gpu_memory_kpi = DurationKpi(
    'flowers_64_Reduce_4_Cards_gpu_memory', 0.1, 0, actived=True)

flowers_64_GPU_1_Cards_train_acc_kpi = AccKpi(
    'flowers_64_GPU_1_Cards_train_acc', 0.03, 0, actived=True)
flowers_64_GPU_1_Cards_train_speed_kpi = AccKpi(
    'flowers_64_GPU_1_Cards_train_speed', 0.06, 0, actived=True)
flowers_64_1_Cards_gpu_memory_kpi = DurationKpi(
    'flowers_64_1_Cards_gpu_memory', 0.1, 0, actived=True)

tracking_kpis = [
    flowers_64_AllReduce_GPU_4_Cards_train_acc_kpi,
    flowers_64_AllReduce_GPU_4_Cards_train_speed_kpi,
    flowers_64_AllReduce_4_Cards_gpu_memory_kpi,
    flowers_64_Reduce_GPU_4_Cards_train_acc_kpi,
    flowers_64_Reduce_GPU_4_Cards_train_speed_kpi,
    flowers_64_Reduce_4_Cards_gpu_memory_kpi,
    flowers_64_GPU_1_Cards_train_acc_kpi,
    flowers_64_GPU_1_Cards_train_speed_kpi,
    flowers_64_1_Cards_gpu_memory_kpi,
]
