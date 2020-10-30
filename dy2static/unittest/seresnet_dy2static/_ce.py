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

static_train_loss_kpi = CostKpi('static_train_loss', 0.002, 0, actived=True)
static_train_acc1_kpi = AccKpi('static_train_acc1', 0.002, 0, actived=True)
static_train_acc5_kpi = AccKpi('static_train_acc5', 0.002, 0, actived=True)
static_train_elapse_kpi = DurationKpi(
    'static_train_elapse', 0.002, 0, actived=True, unit_repr="ms")

dynamic_train_loss_kpi = CostKpi('dynamic_train_loss', 0.002, 0, actived=True)
dynamic_train_acc1_kpi = AccKpi('dynamic_train_acc1', 0.002, 0, actived=True)
dynamic_train_acc5_kpi = AccKpi('dynamic_train_acc5', 0.002, 0, actived=True)
dynamic_train_elapse_kpi = DurationKpi(
    'dynamic_train_elapse', 0.002, 0, actived=True, unit_repr="ms")

tracking_kpis = [
    static_train_loss_kpi, static_train_acc1_kpi, static_train_acc5_kpi,
    static_train_elapse_kpi, dynamic_train_loss_kpi, dynamic_train_acc1_kpi,
    dynamic_train_acc5_kpi, dynamic_train_elapse_kpi
]


def parse_log(log: str):
    """[summary]
    Args:
        log (str): log read from sys std input
    Returns:
        dynamic_logs[-1] (dict): parsed log info
    """
    dynamic_logs = []
    static_logs = []
    for line in log.split('\n'):
        Log = {}
        fs = line.strip().split(',\t')
        try:
            if "ToStatic = False" in fs:
                Log['dynamic_train_loss'] = float(fs[3].split('=')[-1])
                Log['dynamic_train_acc1'] = float(fs[4].split('=')[-1])
                Log['dynamic_train_acc5'] = float(fs[5].split('=')[-1])
                Log['dynamic_train_elapse'] = float(fs[6].split('=')[-1])
                dynamic_logs.append(Log)
            elif "ToStatic = True" in fs:
                Log['static_train_loss'] = float(fs[3].split('=')[-1])
                Log['static_train_acc1'] = float(fs[4].split('=')[-1])
                Log['static_train_acc5'] = float(fs[5].split('=')[-1])
                Log['static_train_elapse'] = float(fs[6].split('=')[-1])
                static_logs.append(Log)
            else:
                pass
        except:
            pass
    return dynamic_logs[-1], static_logs[-1]


def log_to_ce(log: str):
    """[summary]
    Args:
        log (str): log read from sys std input
    """
    final_dynamic_log, final_static_log = parse_log(log)
    print(final_dynamic_log)
    print(final_static_log)

    kpi_tracker = {}
    for kpi in tracking_kpis:
        kpi_tracker[kpi.name] = kpi

    for key in final_dynamic_log:
        kpi_tracker[key].add_record(final_dynamic_log[key])
        kpi_tracker[key].persist()

    for key in final_static_log:
        kpi_tracker[key].add_record(final_static_log[key])
        kpi_tracker[key].persist()


if __name__ == '__main__':
    log = sys.stdin.read()
    print("*****")
    log_to_ce(log)
    print("*****")
