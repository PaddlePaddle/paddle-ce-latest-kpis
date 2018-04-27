"""
continuous_evaluation.py
"""
import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi
from kpi import DurationKpi

lstm_train_cost_kpi = CostKpi('lstm_train_cost', 0.2, 0)
lstm_pass_duration_kpi = DurationKpi('lstm_pass_duration', 0.2, 0)


tracking_kpis = [
    lstm_train_cost_kpi,
    lstm_pass_duration_kpi
]
