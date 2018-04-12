import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

train_acc_kpi = AccKpi('train_acc', 0.05)
test_acc_kpi = AccKpi('test_acc', 0.05)
train_duration_kpi = DurationKpi('train_duration', 0.1)

tracking_kpis = [
    train_acc_kpi,
    test_acc_kpi,
    train_duration_kpi,
]
