"""
continuous_evaluation.py
"""
import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import AccKpi
from kpi import CostKpi
from kpi import DurationKpi

cifar10_128_train_speed_kpi = AccKpi('cifar10_128_train_speed', 0.2, 0)
cifar10_128_gpu_memory_kpi = DurationKpi('cifar10_128_gpu_memory', 0.2, 0)

flowers_32_train_speed_kpi = AccKpi('flowers_32_train_speed', 0.2, 0)
flowers_32_gpu_memory_kpi = DurationKpi('flowers_32_gpu_memory', 0.2, 0)

tracking_kpis = [
    cifar10_128_train_speed_kpi,
    cifar10_128_gpu_memory_kpi,
    flowers_32_train_speed_kpi,
    flowers_32_gpu_memory_kpi,
]
