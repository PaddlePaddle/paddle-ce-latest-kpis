# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.insert(0, os.environ['ceroot'])

from kpi import CostKpi, DurationKpi, AccKpi

dynamic_train_reward_kpi = AccKpi('Average_reward', 0.002, 0, actived=True)
dynamic_train_loss_kpi = CostKpi('loss_probs', 0.002, 0, actived=True)
dynamic_train_elapse_kpi = DurationKpi('Elapse', 0.002, 0, actived=True)

tracking_kpis = [
    dynamic_train_reward_kpi, dynamic_train_loss_kpi, dynamic_train_elapse_kpi
]


def parse_log(log: str):
    """[summary]
    Args:
        log (str): log read from sys std input
    Returns:
        dynamic_logs[-1] (dict): parsed log info
    """
    dynamic_logs = []
    for line in log.split('\n'):
        Log = {}
        fs = line.strip().split(',\t')
        if "ToStatic = False" in fs:
            Log['Average_reward'] = float(fs[3].split('=')[-1])
            Log['loss_probs'] = float(fs[4].split('=')[-1])
            Log['Elapse'] = float(fs[5].split('=')[-1])
            dynamic_logs.append(Log)
        else:
            pass
    return dynamic_logs[-1]


def log_to_ce(log: str):
    """[summary]
    Args:
        log (str): log read from sys std input
    """
    final_dynamic_log = parse_log(log)

    kpi_tracker = {}
    for kpi in tracking_kpis:
        kpi_tracker[kpi.name] = kpi

    for key in final_dynamic_log:
        kpi_tracker[key].add_record(final_dynamic_log[key])
        kpi_tracker[key].persist()


if __name__ == '__main__':
    log = sys.stdin.read()
    print("*****")
    log_to_ce(log)
    print("*****")
