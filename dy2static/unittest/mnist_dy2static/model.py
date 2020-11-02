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

from __future__ import print_function

import argparse
import numpy as np
from time import time

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Linear, Pool2D
from paddle.fluid.optimizer import AdamOptimizer
from paddle.jit import ProgramTranslator

SEED = 2020


class SimpleImgConvPool(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 pool_size,
                 pool_stride,
                 pool_padding=0,
                 pool_type='max',
                 global_pooling=False,
                 conv_stride=1,
                 conv_padding=0,
                 conv_dilation=1,
                 conv_groups=1,
                 act=None,
                 use_cudnn=False,
                 param_attr=None,
                 bias_attr=None):
        super(SimpleImgConvPool, self).__init__()

        self._conv2d = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            param_attr=None,
            bias_attr=None,
            act=act,
            use_cudnn=use_cudnn)

        self._pool2d = Pool2D(
            pool_size=pool_size,
            pool_type=pool_type,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            global_pooling=global_pooling,
            use_cudnn=use_cudnn)

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        return x


class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            1, 20, 5, 2, 2, act="relu")

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            20, 50, 5, 2, 2, act="relu")

        self.pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (self.pool_2_shape**2 * SIZE))**0.5
        self._fc = Linear(
            self.pool_2_shape,
            10,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=scale)),
            act="softmax")

    @paddle.jit.to_static
    def forward(self, inputs, label=None):
        x = self.inference(inputs)
        if label is not None:
            acc = fluid.layers.accuracy(input=x, label=label)
            loss = fluid.layers.cross_entropy(x, label)
            avg_loss = fluid.layers.mean(loss)

            return x, acc, avg_loss
        else:
            return x

    def inference(self, inputs):
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = fluid.layers.reshape(x, shape=[-1, self.pool_2_shape])
        x = self._fc(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser("mnist model benchmark.")
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


def train(args, to_static=False):
    # whether to apply dy2stat
    prog_trans = ProgramTranslator()
    prog_trans.enable(to_static)
    # set device
    device = 'gpu:0' if fluid.is_compiled_with_cuda(
    ) and args.device == 'GPU' else 'cpu'
    paddle.set_device(device)
    # set random seed to initialize parameters
    fluid.default_main_program().random_seed = SEED
    fluid.default_startup_program().random_seed = SEED

    # create model
    mnist = MNIST()
    adam = AdamOptimizer(
        learning_rate=0.001, parameter_list=mnist.parameters())

    # load dataset
    train_dataset = paddle.vision.datasets.MNIST(mode='train', backend='cv2')
    train_loader = paddle.io.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False)

    # start training
    for pass_id in range(args.pass_num):
        # core indicators
        accuracy = []
        loss = []
        cost_time = []
        for batch_id, data in enumerate(train_loader()):
            batch_start = time()

            img = data[0].unsqueeze(1)
            label = data[1]
            prediction, acc, avg_loss = mnist(img, label)

            # backward
            avg_loss.backward()
            adam.minimize(avg_loss)
            mnist.clear_gradients()

            batch_end = time()

            # fetch numpy data
            acc = acc.numpy()[0]
            avg_loss = avg_loss.numpy()[0]
            cost_t = (batch_end - batch_start) * 1000  # ms

            # append data
            accuracy.append(acc)
            loss.append(avg_loss)
            cost_time.append(cost_t)

            if batch_id % 10 == 0:
                print(
                    "ToStatic = %s, Pass = %d, Iter = %d, Loss = %f, Accuracy = %f, Elapse(ms) = %f"
                    % (to_static, pass_id, batch_id, avg_loss, acc, cost_t))
        # print log from each pass_id
        print(
            "to_static=%s, pass=%d, train_avg_acc=%f, train_avg_loss=%f, elapse(ms)=%f"
            % (to_static, pass_id, np.mean(accuracy), np.mean(loss),
               np.mean(cost_time)))


def run_benchmark(args):
    # train in dygraph mode
    train(args, to_static=False)

    # train in static mode
    train(args, to_static=True)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    # train model
    run_benchmark(args)
