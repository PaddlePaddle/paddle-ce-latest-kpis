"""
continuous_evaluation.py
"""
import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi
from kpi import DurationKpi

imikolov_20_avg_ppl_kpi = CostKpi('imikolov_20_avg_ppl', 0.2, 0)
imikolov_20_pass_duration_kpi = DurationKpi('imikolov_20_pass_duration', 0.02,
                                            0, actived=True)
imikolov_20_avg_ppl_kpi_card4 = CostKpi('imikolov_20_avg_ppl_card4', 0.2, 0)
imikolov_20_pass_duration_kpi_card4 = DurationKpi('imikolov_20_pass_duration_card4', 0.03,
                                            0, actived=True)

tracking_kpis = [
    imikolov_20_avg_ppl_kpi,
    imikolov_20_pass_duration_kpi,
    imikolov_20_avg_ppl_kpi_card4,
    imikolov_20_pass_duration_kpi_card4,
]
