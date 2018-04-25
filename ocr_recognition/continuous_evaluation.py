import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

train_avg_loss_kpi = CostKpi('train_avg_loss', 0.05, 0)
train_seq_err_kpi = CostKpi('train_seq_err', 0.05, 0)


tracking_kpis = [
    train_avg_loss_kpi,
    train_seq_err_kpi,
]
