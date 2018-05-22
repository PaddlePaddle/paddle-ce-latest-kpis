"""
continuous_evaluation.py
"""
import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import AccKpi
from kpi import DurationKpi

train_acc_kpi = AccKpi('train_acc', 0.2, 0)
pass_duration_kpi = DurationKpi('pass_duration', 0.02, 0, actived=True)
train_acc_kpi_card4 = AccKpi('train_acc_card4', 0.2, 0)
pass_duration_kpi_card4 = DurationKpi('pass_duration_card4', 0.02, 0, actived=True)

tracking_kpis = [
    train_acc_kpi,
    pass_duration_kpi,
    train_acc_kpi_card4,
    pass_duration_kpi_card4,
]
