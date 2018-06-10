import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

cifar10_train_acc_kpi = AccKpi('cifar10_train_acc', 0.02, 0, actived=True)
cifar10_train_speed_kpi = AccKpi('cifar10_train_speed', 0.02, 0, actived=True)
cifar10_gpu_memory_kpi = DurationKpi('cifar10_gpu_memory', 0.02, 0)
flowers_train_speed_kpi = AccKpi('flowers_train_speed', 0.02, 0, actived=True)
flowers_gpu_memory_kpi = DurationKpi('flowers_gpu_memory', 0.02, 0)

cifar10_train_acc_kpi_card4 = AccKpi('cifar10_train_acc_card4', 0.02, 0, actived=True)
cifar10_train_speed_kpi_card4 = AccKpi('cifar10_train_speed_card4', 0.02, 0, actived=True)
cifar10_gpu_memory_kpi_card4 = DurationKpi('cifar10_gpu_memory_card4', 0.02, 0)
flowers_train_speed_kpi_card4 = AccKpi('flowers_train_speed_card4', 0.02, 0, actived=True)
flowers_gpu_memory_kpi_card4 = DurationKpi('flowers_gpu_memory_card4', 0.02, 0)

cifar10_train_acc_kpi_card8 = AccKpi('cifar10_train_acc_card8', 0.02, 0, actived=True)
cifar10_train_speed_kpi_card8 = AccKpi('cifar10_train_speed_card8', 0.02, 0, actived=True)
cifar10_gpu_memory_kpi_card8 = DurationKpi('cifar10_gpu_memory_card8', 0.02, 0, actived=True)
flowers_train_speed_kpi_card8 = AccKpi('flowers_train_speed_card8', 0.02, 0, actived=True)
flowers_gpu_memory_kpi_card8 = DurationKpi('flowers_gpu_memory_card8', 0.02, 0, actived=True)

tracking_kpis = [
    cifar10_train_acc_kpi,
    cifar10_train_speed_kpi,
    cifar10_gpu_memory_kpi,
    flowers_train_speed_kpi,
    flowers_gpu_memory_kpi,
    
    cifar10_train_acc_kpi_card4,
    cifar10_train_speed_kpi_card4,
    cifar10_gpu_memory_kpi_card4,
    flowers_train_speed_kpi_card4,
    flowers_gpu_memory_kpi_card4,
    
    cifar10_train_acc_kpi_card8,
    cifar10_train_speed_kpi_card8,
    cifar10_gpu_memory_kpi_card8,
    flowers_train_speed_kpi_card8,
    flowers_gpu_memory_kpi_card8,
]
