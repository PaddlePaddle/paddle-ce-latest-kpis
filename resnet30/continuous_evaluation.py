import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi

train_cost_kpi = CostKpi('train_cost', 0.05)
train_duration_kpi = DurationKpi('train_duration', 0.05)

tracking_kpis = [
    train_cost_kpi,
    train_duration_kpi,
]
