# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import os
import re

from continuous_evaluation import tracking_kpis

train_speed_kpi = tracking_kpis[0]
train_acc_kpi = tracking_kpis[1]

with open('./training_result', 'r') as f:
    lines = f.readlines()

# persist train speed kpi
training_speed = []
for line in lines:
    train_speed_str_pattern = re.compile('Total examples')
    m = train_speed_str_pattern.match(line)
    if m:
        pattern = re.compile('\d+\.\d+')
        m = pattern.findall(line)
        training_speed.append(float(m[1]))

training_speed = np.mean(training_speed)
train_speed_kpi.add_record(np.array(training_speed, dtype='float32'))
print("train_speed_kpi: ", training_speed)
train_speed_kpi.persist()

# persist train acc kpi
for line in lines:
    train_acc_str_pattern = re.compile('Pass')
    m = train_acc_str_pattern.match(line)
    if m:
        pattern = re.compile('\d+\.\d+')
        m = pattern.findall(line)
train_acc = float(m[0])

train_acc_kpi.add_record(np.array(train_acc, dtype='float32'))
print("train_acc_kpi:", train_acc)
train_acc_kpi.persist()
