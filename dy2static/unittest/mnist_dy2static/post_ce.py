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
import json
import logging

import numpy as np

sys.path.append("../../utils")

from ce_post_tools import post_data
from ce_post_tools import load_records_from

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

KpiThreshold={}
KpiThreshold['static_train_avg_acc'] = 0.002
KpiThreshold['static_train_avg_loss'] = 0.002
KpiThreshold['static_train_elapse'] = 0.002

KpiThreshold['dynamic_train_avg_acc'] = 0.002
KpiThreshold['dynamic_train_avg_loss'] = 0.002
KpiThreshold['dynamic_train_elapse'] = 0.002


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

def log2kpis(dict_logs: dict) -> list:
    """
    convert dict log to kpi jsons
    Args:
        dict_logs (dict): [description]
    Returns:
        list: [description]
    """
    dict_kpis = []
    model_path = os.path.abspath(os.path.dirname(__file__))
    model_name = model_path.split("/")[-1]
    for key, value in dict_logs.items():
        dict_kpi = {}
        kpi_file = "./latest_kpis/{}_factor.txt".format(key)
        kpi_base = float(load_records_from(kpi_file))
    
        # kpi value calculation
        kpi_ratio = 0 # default value
        if "acc1" in key.split('_') or "acc5" in key.split('_'):
            # acc kpis, LessWorseKpi
            kpi_ratio = value - kpi_base / kpi_base
        else:
            # loss kpis, time elapse kpis, GreaterWorseKpi
            kpi_ratio = kpi_base - value / kpi_base
        kpi_status = "Passed" if (-kpi_ratio) < KpiThreshold[key] else "Failed"

        dict_kpi["model_name"] = model_name
        dict_kpi["kpi_name"] = key
        dict_kpi["kpi_status"] = kpi_status
        dict_kpi["kpi_base"] = kpi_base
        dict_kpi["kpi_value"] = value
        dict_kpi["threshold"] = KpiThreshold[key]
        dict_kpi["ratio"] = round(kpi_ratio, 3)
        dict_kpis.append(dict_kpi)
    return dict_kpis

def main():
    """
    main
    """
    log = sys.stdin.read()
    logger.info("==== load log succeed ====")
    dict_log = parse_log(log)
    logger.info("==== parse log succeed ====")
    dict_kpis = log2kpis(dict_log)
    logger.info("==== convert log succeed ====")
    post_data(dict_kpis)
    logger.info("==== post log succeed ====")

if __name__ == "__main__":
    main()



