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

tracking_kpis = [
    imikolov_20_avg_ppl_kpi,
    imikolov_20_pass_duration_kpi,
]
