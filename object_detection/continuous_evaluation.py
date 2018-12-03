import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

train_cost_kpi = CostKpi('train_cost', 0.02, 0, actived=True)
train_speed_kpi = AccKpi('train_speed', 0.03, 0, actived=True)
four_card_speed_kpi = AccKpi('four_card_train_speed', 0.03, 0, actived=True)

tracking_kpis = [train_cost_kpi, train_speed_kpi, four_card_speed_kpi]
