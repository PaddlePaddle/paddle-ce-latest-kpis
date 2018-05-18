import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

train_acc_top1_kpi = AccKpi('train_acc_top1_kpi', 0.05, 0,
                            actived=True,
                            desc='TOP1 ACC')
train_acc_top5_kpi = AccKpi('train_acc_top5_kpi', 0.05, 0,
                            actived=True,
                            desc='TOP5 ACC')
train_cost_kpi = CostKpi('train_cost_kpi', 0.05, 0,
                        actived=True,
                        desc='train cost')
train_speed_kpi = AccKpi('train_speed_kpi', 0.05, 0,
                        actived=True,
                        unit_repr='images/s',
                        desc='train speed in one GPU card')
four_card_train_speed_kpi = AccKpi('four_card_train_speed_kpi', 0.05, 0,
                        actived=True,
                        unit_repr='images/s',
                        desc='train speed in four GPU card')

tracking_kpis = [train_acc_top1_kpi,
                train_acc_top5_kpi,
                train_cost_kpi,
                train_speed_kpi,
                four_card_train_speed_kpi]
