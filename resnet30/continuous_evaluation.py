import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, AccKpi, DurationKpi

train_cost_kpi = CostKpi('train_cost', 0.05, actived=True)
train_acc_kpi = AccKpi('train_acc', 0.02, actived=True)  
test_acc_kpi = AccKpi('test_acc', 0.05, actived=True)  
train_speed_kpi = AccKpi('train_speed', 0.01, actived=True)  
train_duration_kpi = DurationKpi('train_duration', 0.02, actived=True)


tracking_kpis = [
    train_cost_kpi,
    train_acc_kpi,
    test_acc_kpi,
    train_speed_kpi,
    train_duration_kpi,
]
