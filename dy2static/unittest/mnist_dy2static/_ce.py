#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

static_train_acc_kpi = AccKpi('static_train_avg_acc', 0.002, 0, actived=True)
static_train_loss_kpi = CostKpi(
    'static_train_avg_loss', 0.002, 0, actived=True)
static_train_elapse_kpi = DurationKpi(
    'static_train_elapse', 0.002, 0, actived=True, unit_repr="ms")

dynamic_train_acc_kpi = AccKpi('dynamic_train_avg_acc', 0.002, 0, actived=True)
dynamic_train_loss_kpi = CostKpi(
    'dynamic_train_avg_loss', 0.002, 0, actived=True)
dynamic_train_elapse_kpi = DurationKpi(
    'dynamic_train_elapse', 0.002, 0, actived=True, unit_repr="ms")

tracking_kpis = [
    static_train_acc_kpi, static_train_loss_kpi, static_train_elapse_kpi,
    dynamic_train_acc_kpi, dynamic_train_loss_kpi, dynamic_train_elapse_kpi
]


def parse_log(log: str):
    """[summary]
    Args:
        log (str): log read from sys std input
    Returns:
        static_logs[-1] (dict): parsed log info
    """
    dy2staic_elapse = []
    dynamic_elapse = []
    for line in log.split('\n'):
        Log = {}
        fs = line.strip().split(', ')
        if "to_static=True" in fs:
            '''
            after split, sample list is as below
            fs = ['to_static=True',
                  'pass=0',
                  'train_avg_acc=0.895750',
                  'train_avg_loss=0.843321',
                  'elapse(ms)=11.286300']
            '''
            static_train_avg_acc = float(fs[2].split('=')[-1])
            static_train_avg_loss = float(fs[3].split('=')[-1])
            dy2staic_elapse.append(float(fs[4].split('=')[-1]))
        elif "to_static=False" in fs:
            dynamic_train_avg_acc = float(fs[2].split('=')[-1])
            dynamic_train_avg_loss = float(fs[3].split('=')[-1])
            dynamic_elapse.append(float(fs[4].split('=')[-1]))
        else:
            pass

    Log['static_train_avg_acc'] = static_train_avg_acc
    Log['static_train_avg_loss'] = static_train_avg_loss
    Log['static_train_elapse'] = sum(dy2staic_elapse) / len(dy2staic_elapse)

    Log['dynamic_train_avg_acc'] = dynamic_train_avg_acc
    Log['dynamic_train_avg_loss'] = dynamic_train_avg_loss
    Log['dynamic_train_elapse'] = sum(dynamic_elapse) / len(dynamic_elapse)
    return Log


def log_to_ce(log: str):
    """[summary]
    Args:
        log (str): log read from sys std input
    """
    dict_log = parse_log(log)

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
