import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

train_cost_kpi = CostKpi('train_cost', 0.02, actived=True)
test_acc_kpi = AccKpi('test_acc', 0.005, actived=True)
train_duration_kpi = DurationKpi('train_duration', 0.02, actived=True)
train_acc_kpi = AccKpi('train_acc', 0.005, actived=True)

tracking_kpis = [
    train_acc_kpi,
    train_cost_kpi,
    test_acc_kpi,
    train_duration_kpi,
]
