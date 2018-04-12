"""
continuous_evaluation.py
"""
import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import AccKpi
from kpi import CostKpi
from kpi import DurationKpi

wmb_128_train_speed_kpi = AccKpi('wmb_128_train_speed', 0.15, 0)
wmb_128_gpu_memory_kpi = DurationKpi('wmb_128_gpu_memory', 0.01, 0)


tracking_kpis = [
    wmb_128_train_speed_kpi,
    wmb_128_gpu_memory_kpi,
]
