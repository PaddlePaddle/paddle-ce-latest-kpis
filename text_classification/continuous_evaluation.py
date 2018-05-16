"""
continuous_evaluation.py
"""
import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi
from kpi import DurationKpi

lstm_train_cost_kpi = CostKpi('lstm_train_cost', 5, 0)
lstm_pass_duration_kpi = DurationKpi('lstm_pass_duration', 0.02, 0, actived=True)

lstm_train_cost_kpi_card4 = CostKpi('lstm_train_cost_card4', 0.2, 0)
lstm_pass_duration_kpi_card4 = DurationKpi('lstm_pass_duration_card4', 0.05, 0, actived=True)

tracking_kpis = [
              lstm_train_cost_kpi, lstm_pass_duration_kpi,
              lstm_train_cost_kpi_card4, lstm_pass_duration_kpi_card4,
                ]
