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
KpiThreshold['dynamic_train_loss'] = 0.002
KpiThreshold['dynamic_train_acc1'] = 0.002
KpiThreshold['dynamic_train_acc5'] = 0.002
KpiThreshold['dynamic_train_elapse'] = 0.002
KpiThreshold['static_train_loss'] = 0.002
KpiThreshold['static_train_acc1'] = 0.002
KpiThreshold['static_train_acc5'] = 0.002
KpiThreshold['static_train_elapse'] = 0.002


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
        kpi_ratio = abs(value - kpi_base) / kpi_base

        if "acc1" in key.split('_') or "acc5" in key.split('_'):
            kpi_status = "Passed" if kpi_ratio >= KpiThreshold[key] else "Failed"
        else:
            kpi_status = "Passed" if kpi_ratio <= KpiThreshold[key] else "Failed"

        dict_kpi["model_name"] = model_name
        dict_kpi["kpi_name"] = key
        dict_kpi["kpi_status"] = kpi_status
        dict_kpi["kpi_base"] = kpi_base
        dict_kpi["kpi_value"] = value
        dict_kpi["threshold"] = KpiThreshold[key]
        dict_kpi["ratio"] = round(kpi_ratio, 2)
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



