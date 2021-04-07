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
import time
import logging

import requests
import numpy as np


FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

def load_records_from(file):
    '''
    each line of the data format is
        <json of record>
    for example, a real record might be:
        [[0.1, 0.3], [0.4, 0.2]]
    '''
    datas = []
    with open(file) as f:
        for line in f.readlines():
            data = json.loads(line.strip())
            datas.append(np.array(data))
    return np.array(datas)


def check_environ(env_list: list):
    """
    check environment variable exist or not
    Args:
        env_list (list): list of enviroment variables' name need to check
    """
    for env_name in env_list:
        assert os.environ.get(env_name) is not None, "===> ${} is not export".format(env_name)


def get_commit_id(description_path : str) -> dict:
    """
    parse commit id from description.txt
    Args:
        description_path (str): [description]
    Returns:
        dict: [description]
    """
    assert description_path is not None, "===> description_path cannot be empty"
    result = {}
    try:
        f = open(description_path)
        for line in f.readlines():
            line = line.strip()
            if line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                result[key] = value
    except Exception as e:
        logger.info("==== parse {} failed ====".format(description_path))
        logger.error(e)
    return result


def send_data(dict_kpis, description_path, url=None):
    """
    send json data to backend
    Args:
        dict_kpis ([type]): [description]
        description_path ([type]): [description]
        url ([type], optional): [description]. Defaults to None.
    """
    assert url is not None, "===> url cannot be empty"
    description_info = get_commit_id(description_path)

    params = {
        "build_type": os.environ["build_type"],
        "task_name": os.environ["task_name"],
        "owner": os.environ["owner"],
        "build_id": os.environ["build_id"],
        "build_number": os.environ["build_number"],
        "commit_id": description_info["commit_id"],
        "repo": description_info["repo"],
        "branch": description_info["branch"],
        "create_time": int(time.time()),
        "duration": None,
        "case_detail": json.dumps(dict_kpis)
    }
    response = requests.post(url, data=params)
    logger.info(response)
    assert response.status_code == 200, "send post request failed, please check input data structure and post urls"
    logger.info("==== post request succeed, response status_code is {} ====".
                format(response.status_code))


def post_data(dict_kpis):
    """
    post data
    Args:
        dict_kpis ([type]): [description]
    """
    basic_envrion = ["build_type", "task_name", "owner",
                     "build_id", "build_number", "post_url"]
    check_environ(basic_envrion)
    description_path = os.environ.get("description_path",
                                      "/workspace/ce/new_ce/src/artifacts")

    send_data(dict_kpis,
              description_path=description_path,
              url=os.environ["post_url"])


if __name__ == "__main__":
    # for test
    dict_kpis = [{
        "model_name": "test_dy2static",
        "kpi_name": "dynamic_train_loss",
        "kpi_status": "Passed",
        "kpi_base": 0.01,
        "kpi_value": 0.02,
        "threshold": 0.05,
        "ratio": 0.1
            }]
    post_data(dict_kpis)

