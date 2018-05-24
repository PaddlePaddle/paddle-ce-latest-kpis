import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import LessWorseKpi

speedup_rate_kpi = LessWorseKpi('speedup_rate', 0.01)

tracking_kpis = [
    speedup_rate_kpi,
]
