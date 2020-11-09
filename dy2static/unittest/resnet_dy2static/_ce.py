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
    dynamic_train_elapse = []
    static_train_elapse = []
    for line in log.split('\n'):
        Log = {}
        fs = line.strip().split(',\t')
        try:
            if "ToStatic = False" in fs:
                dynamic_train_loss = float(fs[3].split('=')[-1])
                dynamic_train_acc1 = float(fs[4].split('=')[-1])
                dynamic_train_acc5 = float(fs[5].split('=')[-1])
                dynamic_train_elapse.append(float(fs[6].split('=')[-1]))
            elif "ToStatic = True" in fs:
                static_train_loss = float(fs[3].split('=')[-1])
                static_train_acc1 = float(fs[4].split('=')[-1])
                static_train_acc5 = float(fs[5].split('=')[-1])
                static_train_elapse.append(float(fs[6].split('=')[-1]))
            else:
                pass
        except:
            pass

    Log['dynamic_train_loss'] = dynamic_train_loss
    Log['dynamic_train_acc1'] = dynamic_train_acc1
    Log['dynamic_train_acc5'] = dynamic_train_acc5
    Log['dynamic_train_elapse'] = sum(dynamic_train_elapse) / len(
        dynamic_train_elapse)

    Log['static_train_loss'] = static_train_loss
    Log['static_train_acc1'] = static_train_acc1
    Log['static_train_acc5'] = static_train_acc5
    Log['static_train_elapse'] = sum(static_train_elapse) / len(
        static_train_elapse)
    return Log


def log_to_ce(log: str):
    """[summary]
    Args:
        log (str): log read from sys std input
    """
    dict_log = parse_log(log)
    print(dict_log)

    kpi_tracker = {}
    for kpi in tracking_kpis:
        kpi_tracker[kpi.name] = kpi

    for key in dict_log:
        kpi_tracker[key].add_record(dict_log[key])
        kpi_tracker[key].persist()


if __name__ == '__main__':
    log = sys.stdin.read()
    print("*****")
    log_to_ce(log)
    print("*****")
