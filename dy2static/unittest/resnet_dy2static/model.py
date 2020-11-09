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

from __future__ import print_function

import math
import time
import argparse
import numpy as np

import paddle

SEED = 2020

if paddle.is_compiled_with_cuda():
    paddle.fluid.set_flags({'FLAGS_cudnn_deterministic': True})

program_translator = paddle.jit.ProgramTranslator()



place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
    else paddle.CPUPlace()


def parse_args():
    parser = argparse.ArgumentParser("ResNet model benchmark.")
    # parser.add_argument(
    #     '--to_static', type=bool, default=True, help='whether to train model in static mode.')
    parser.add_argument(
        '--batch_size', type=int, default=64, help='The minibatch size.')
    parser.add_argument(
        '--pass_num', type=int, default=20, help='The number of passes.')
    parser.add_argument(
        '--base_lr',
        type=float,
        default=0.001,
        help='The basic learning rate.')
    parser.add_argument(
        '--momentum_rate',
        type=float,
        default=0.9,
        help='The momentum rate of optimizer.')
    parser.add_argument(
        '--l2_decay', type=float, default=1e-4, help='L2 decay value.')
    parser.add_argument(
        '--log_internal',
        type=int,
        default=10,
        help='The internal step of log.')
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


def optimizer_setting(args, parameter_list=None):
    optimizer = paddle.optimizer.Momentum(
        learning_rate=args.base_lr,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay),
        parameters=parameter_list)

    return optimizer


class ConvBNLayer(paddle.nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self._conv = paddle.nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias_attr=False)

        self._batch_norm = paddle.nn.BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y


class BottleneckBlock(paddle.nn.Layer):
    def __init__(self, num_channels, num_filters, stride, shortcut=True):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(x=short, y=conv2)

        layer_helper = paddle.fluid.layer_helper.LayerHelper(
            self.full_name(), act='relu')
        return layer_helper.append_activation(y)


class ResNet(paddle.nn.Layer):
    def __init__(self, layers=50, class_dim=102):
        super(ResNet, self).__init__()

        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_channels = [64, 256, 512, 1024]
        num_filters = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        self.pool2d_max = paddle.nn.Pool2D(
            pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

        self.bottleneck_block_list = []
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels[block]
                        if i == 0 else num_filters[block] * 4,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut))
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = paddle.fluid.dygraph.Pool2D(
            pool_size=7, pool_type='avg', global_pooling=True)

        self.pool2d_avg_output = num_filters[len(num_filters) - 1] * 4 * 1 * 1

        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.out = paddle.nn.Linear(
            in_features=self.pool2d_avg_output,
            out_features=class_dim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-stdv, stdv)))

    @paddle.jit.to_static
    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = paddle.reshape(y, shape=[-1, self.pool2d_avg_output])
        pred = self.out(y)
        pred = paddle.nn.functional.softmax(pred)

        return pred


def reader_decorator(reader):
    def __reader__():
        for item in reader():
            img = np.array(item[0]).astype('float32').reshape(3, 224, 224)
            label = np.array(item[1]).astype('int64').reshape(1)
            yield img, label

    return __reader__


def train(args, to_static):
    # set random seed
    np.random.seed(SEED)
    paddle.seed(SEED)
    paddle.framework.random._manual_program_seed(SEED)
    # set device
    device = 'gpu:0' if paddle.fluid.is_compiled_with_cuda(
    ) and args.device == 'GPU' else 'cpu'
    paddle.set_device(device)

    # load flowers data
    train_reader = paddle.batch(
        reader_decorator(paddle.dataset.flowers.train(use_xmap=False)),
        batch_size=args.batch_size,
        drop_last=True)
    data_loader = paddle.io.DataLoader.from_generator(
        capacity=5, iterable=True)
    data_loader.set_sample_list_generator(train_reader)

    # create model
    resnet = ResNet()
    optimizer = optimizer_setting(args, parameter_list=resnet.parameters())

    # start training
    for pass_id in range(args.pass_num):
        total_loss = 0.0
        total_acc1 = 0.0
        total_acc5 = 0.0
        total_sample = 0
        cost_time = 0.

        for batch_id, data in enumerate(data_loader()):
            start_time = time.time()
            img, label = data

            pred = resnet(img)
            loss = paddle.nn.functional.cross_entropy(input=pred, label=label)
            avg_loss = paddle.mean(x=loss)
            acc_top1 = paddle.metric.accuracy(input=pred, label=label, k=1)
            acc_top5 = paddle.metric.accuracy(input=pred, label=label, k=5)

            # backward
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            resnet.clear_gradients()

            # cost time
            end_time = time.time()
            cost_time += (end_time - start_time) * 1000  # ms

            # append data of core indicators
            total_loss += avg_loss
            total_acc1 += acc_top1
            total_acc5 += acc_top5
            total_sample += 1

            if batch_id % args.log_internal == 0:
                print( "ToStatic = {},\tPass = {},\tIter = {},\tLoss = {:.3f},\tAcc1 = {:.3f},\tAcc5 = {:.3f},\tElapse(ms) = {:.3f}".format
                    ( to_static, pass_id, batch_id, total_loss.numpy()[0] / total_sample, \
                        total_acc1.numpy()[0] / total_sample, total_acc5.numpy()[0] / total_sample, cost_time / args.log_internal))
                # reset cost_time
                cost_time = 0.

    return total_loss.numpy()


def export_inference_model(class_dim=1000):
    paddle.set_device('cpu')
    for layers_num in [50, 101]:
        resnet = ResNet(layers=layers_num, class_dim=class_dim)

        img = paddle.randn([1, 3, 224, 224])
        pred = resnet(img)
        paddle.jit.save(resnet, 'resnet_{}/x'.format(layers_num))


def run_benchmark(args):
    # train in dygraph mode
    train(args, to_static=False)

    # train in static mode
    train(args, to_static=True)

    # save inference model with class_dim=1000
    export_inference_model()


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    # train model
    run_benchmark(args)
