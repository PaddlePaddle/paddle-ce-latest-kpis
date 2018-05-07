"""
continuous_evaluation.py
"""
import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import AccKpi
from kpi import DurationKpi

train_acc_kpi = AccKpi('train_acc', 0.2, 0)
pass_duration_kpi = DurationKpi('pass_duration', 0.2, 0)

tracking_kpis = [
    train_acc_kpi,
    pass_duration_kpi,
]
