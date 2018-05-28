import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

train_avg_ppl_kpi = CostKpi('train_avg_ppl_kpi', 0.2, 0)
train_pass_duration_kpi = DurationKpi('train_pass_duration_kpi', 0.2, 0)

tracking_kpis = [
    train_avg_ppl_kpi,
    train_pass_duration_kpi,
]
