#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import download


def build_vocab(filename):
    EOS = "</eos>"
    vocab_dict = {}
    ids = 0
    vocab_dict[EOS] = ids
    ids += 1

    with open(filename, "r") as f:
        for line in f.readlines():
            for w in line.strip().split():
                if w not in vocab_dict:
                    vocab_dict[w] = ids
                    ids += 1
    print("vocab word num", ids)
    return vocab_dict


def file_to_ids(src_file, src_vocab):
    src_data = []
    with open(src_file, "r") as f_src:
        for line in f_src.readlines():
            arra = line.strip().split()
            ids = [src_vocab[w] for w in arra if w in src_vocab]

            src_data += ids + [0]
    return src_data


def get_ptb_data(batch_size, num_steps):
    data_path = '/root/.cache/paddle/dataset/ptb_lm'
    flag_value = 1342
    download.get_uncompress_data(data_path, flag_value)

    train_file = os.path.join(data_path, 'simple-examples', 'data',
                              'ptb.train.txt')
    vocab_dict = build_vocab(train_file)
    train_ids = file_to_ids(train_file, vocab_dict)

    data_len = len(train_ids)
    raw_data = np.asarray(train_ids, dtype="int64")
    batch_len = data_len // batch_size
    data = raw_data[0:batch_size * batch_len].reshape((batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps

    data_ret = [[
        np.copy(data[:, i * num_steps:(i + 1) * num_steps]),
        np.copy(data[:, i * num_steps + 1:(i + 1) * num_steps + 1])
    ] for i in range(epoch_size)]
    return data_ret
