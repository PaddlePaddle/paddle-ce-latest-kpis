"""
continuous_evaluation.py
"""
import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import AccKpi
from kpi import CostKpi
from kpi import DurationKpi

imdb_32_train_speed_kpi = AccKpi('imdb_32_train_speed', 0.15, 0)
imdb_32_gpu_memory_kpi = DurationKpi('imdb_32_gpu_memory', 0.01, 0)


tracking_kpis = [
    imdb_32_train_speed_kpi,
    imdb_32_gpu_memory_kpi,
]
