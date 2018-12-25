import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

test_avg_ppl_kpi = CostKpi('test_avg_ppl_kpi', 0.2, 0)
train_pass_duration_kpi = DurationKpi('train_pass_duration_kpi', 0.03, 0, actived=True)
test_avg_ppl_kpi_card4 = CostKpi('test_avg_ppl_kpi_card4', 0.05, 0, actived=True)
train_pass_duration_kpi_card4 = DurationKpi('train_pass_duration_kpi_card4', 0.05, 0, actived=True)

tracking_kpis = [
    test_avg_ppl_kpi,
    train_pass_duration_kpi,
    test_avg_ppl_kpi_card4,
    train_pass_duration_kpi_card4,
]
