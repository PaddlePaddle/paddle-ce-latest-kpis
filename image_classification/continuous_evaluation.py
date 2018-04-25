import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

train_acc_top1_kpi = AccKpi('train_acc_top1_kpi', 0.1, 0)
train_acc_top5_kpi = AccKpi('train_acc_top5_kpi', 0.1, 0)
train_cost_kpi = CostKpi('train_cost_kpi', 0.1, 0)


tracking_kpis = [
	train_acc_top1_kpi,
	train_acc_top5_kpi,
	train_cost_kpi
]
