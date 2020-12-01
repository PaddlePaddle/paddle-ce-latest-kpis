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

import paddle.fluid as fluid
import os
import io
import sys
import random
import numpy as np
import download


def get_vocab(file_path):
    vocab = {}
    wid = 0
    with io.open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            if line.strip() not in vocab:
                vocab[line.strip()] = wid
                wid += 1
    vocab["<unk>"] = len(vocab)
    return vocab, wid + 1


def get_train_data_generator(batch_size, epoch, max_seq_len, shuffle=True):
    home_path = os.path.expanduser('~')
    data_path = os.path.join(home_path, '.cache/paddle/dataset/sentiment')
    flag_value = 1342
    download.get_uncompress_data(data_path, flag_value)

    def data_reader():
        #  load vocab
        vocab_path = os.path.join(data_path, 'senta_data', 'word_dict.txt')
        word_dict, unk_id = get_vocab(vocab_path)
        #  load dataset
        pad_id = 0
        all_data = []
        file_path = os.path.join(data_path, 'senta_data', 'train.tsv')
        print(file_path)
        with io.open(file_path, "r", encoding='utf8') as fin:
            for line in fin:
                if line.startswith('text_a'):
                    continue
                cols = line.strip().split('\t')
                if len(cols) != 2:
                    sys.stderr.write("[NOTICE] Error Format Line!")
                    continue
                label = [int(cols[1])]
                wids = [
                    word_dict[x] if x in word_dict else unk_id
                    for x in cols[0].split(" ")
                ]
                seq_len = len(wids)
                if seq_len < max_seq_len:
                    for i in range(max_seq_len - seq_len):
                        wids.append(pad_id)
                else:
                    wids = wids[:max_seq_len]
                    seq_len = max_seq_len
                all_data.append((wids, label, seq_len))
        if shuffle:
            random.shuffle(all_data)
        #num_examples[phrase] = len(all_data)

        def reader():
            for epoch_index in range(epoch):
                for doc, label, seq_len in all_data:
                    yield doc, label, seq_len

        return reader

    return fluid.io.batch(data_reader(), batch_size)
