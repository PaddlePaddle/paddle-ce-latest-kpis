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

import time
import argparse

import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable, ProgramTranslator

from senta_net import CNN, GRU, BOW, BiGRU
import getdata

SEED = 2020
program_translator = ProgramTranslator()

# Note: Set True to eliminate randomness.
#     1. For one operation, cuDNN has several algorithms,
#        some algorithm results are non-deterministic, like convolution algorithms.
if fluid.is_compiled_with_cuda():
    fluid.set_flags({'FLAGS_cudnn_deterministic': True})


def parse_args():
    parser = argparse.ArgumentParser("sentiment model benchmark.")
    # parser.add_argument(
    #     '--to_static', type=bool, default=True, help='whether to train model in static mode.')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='The minibatch size.')
    parser.add_argument(
        '--pass_num', type=int, default=5, help='The number of passes.')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def add_args(args):
    #batch_size = 4
    args.class_num = 2
    args.lr = 0.01
    args.vocab_size = 33256
    args.max_seq_len = 256
    args.padding_size = 50
    args.log_step = 5
    args.train_step = 50
    args.model_type = 'gru_net'
    return args


def train(args, to_static):
    program_translator.enable(to_static)
    place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() \
        else fluid.CPUPlace()

    with fluid.dygraph.guard(place):
        np.random.seed(SEED)
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

        train_reader = getdata.get_train_data_generator(
            args.batch_size, args.pass_num, args.max_seq_len)
        train_loader = fluid.io.DataLoader.from_generator(capacity=24)
        train_loader.set_sample_list_generator(train_reader)

        if args.model_type == 'cnn_net':
            model = CNN(args.vocab_size, args.batch_size, args.padding_size)
        elif args.model_type == 'bow_net':
            model = BOW(args.vocab_size, args.batch_size, args.padding_size)
        elif args.model_type == 'gru_net':
            model = GRU(args.vocab_size, args.batch_size, args.padding_size)
        elif args.model_type == 'bigru_net':
            model = BiGRU(args.vocab_size, args.batch_size, args.padding_size)
        sgd_optimizer = fluid.optimizer.Adagrad(
            learning_rate=args.lr, parameter_list=model.parameters())

        #loss_data = []
        total_loss = 0.0
        total_acc = 0.0
        for pass_id in range(args.pass_num):
            cost_time = 0
            total_sample = 0
            for batch_id, data in enumerate(train_loader()):
                start_time = time.time()
                word_ids, labels, seq_lens = data
                doc = to_variable(word_ids.numpy().reshape(-1)).astype('int64')
                label = labels.astype('int64')

                model.train()
                avg_cost, prediction, acc = model(doc, label)
                #loss_data.append(avg_cost.numpy()[0])

                avg_cost.backward()
                sgd_optimizer.minimize(avg_cost)
                model.clear_gradients()

                cost_time += (time.time() - start_time) * 1000  # ms

                total_loss += avg_cost
                total_acc += acc
                total_sample += 1

                if (batch_id + 1) % args.log_step == 0:
                    #time_end = time.time()
                    #used_time = time_end - time_begin
                    print(
                        "ToStatic = {},\tPass = {},\tIter = {},\tLoss = {:.3f},\tAcc = {:.3f},\tElapse(ms) = {:.3f}"
                        .format(to_static, pass_id, batch_id,
                                total_loss.numpy()[0] / total_sample,
                                total_acc.numpy()[0] / total_sample, cost_time
                                / args.log_step))
                    cost_time = 0
                    #time_begin = time.time()

                if batch_id == args.train_step:
                    break
                batch_id += 1
    return total_loss


def run_benchmark(args):
    # train in dygraph mode
    args = add_args(args)
    print('dygraph mode')
    train(args, to_static=False)

    print('static mode')
    # train in static mode
    train(args, to_static=True)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    # train model
    run_benchmark(args)
