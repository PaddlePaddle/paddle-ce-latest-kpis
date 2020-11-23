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
import numpy as np
import argparse
import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.jit import to_static, ProgramTranslator

# Note: Set True to eliminate randomness.
#     1. For one operation, cuDNN has several algorithms,
#        some algorithm results are non-deterministic, like convolution algorithms.
if paddle.is_compiled_with_cuda():
    paddle.fluid.set_flags({'FLAGS_cudnn_deterministic': True})

SEED = 2020
program_translator = ProgramTranslator()


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 channels=None,
                 num_groups=1,
                 act='relu',
                 use_cudnn=True,
                 name=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(
                initializer=MSRA(), name=self.full_name() + "_weights"),
            bias_attr=False)

        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name=self.full_name() + "_bn" + "_scale"),
            bias_attr=ParamAttr(name=self.full_name() + "_bn" + "_offset"),
            moving_mean_name=self.full_name() + "_bn" + '_mean',
            moving_variance_name=self.full_name() + "_bn" + '_variance')

    def forward(self, inputs, if_act=False):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if if_act:
            y = fluid.layers.relu6(y)
        return y


class DepthwiseSeparable(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters1,
                 num_filters2,
                 num_groups,
                 stride,
                 scale,
                 name=None):
        super(DepthwiseSeparable, self).__init__()

        self._depthwise_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=int(num_filters1 * scale),
            filter_size=3,
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            use_cudnn=True)

        self._pointwise_conv = ConvBNLayer(
            num_channels=int(num_filters1 * scale),
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0)

    def forward(self, inputs):
        y = self._depthwise_conv(inputs)
        y = self._pointwise_conv(y)
        return y


class MobileNetV1(fluid.dygraph.Layer):
    def __init__(self, scale=1.0, class_dim=1000):
        super(MobileNetV1, self).__init__()
        self.scale = scale
        self.dwsl = []

        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            channels=3,
            num_filters=int(32 * scale),
            stride=2,
            padding=1)

        dws21 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(32 * scale),
                num_filters1=32,
                num_filters2=64,
                num_groups=32,
                stride=1,
                scale=scale),
            name="conv2_1")
        self.dwsl.append(dws21)

        dws22 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(64 * scale),
                num_filters1=64,
                num_filters2=128,
                num_groups=64,
                stride=2,
                scale=scale),
            name="conv2_2")
        self.dwsl.append(dws22)

        dws31 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(128 * scale),
                num_filters1=128,
                num_filters2=128,
                num_groups=128,
                stride=1,
                scale=scale),
            name="conv3_1")
        self.dwsl.append(dws31)

        dws32 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(128 * scale),
                num_filters1=128,
                num_filters2=256,
                num_groups=128,
                stride=2,
                scale=scale),
            name="conv3_2")
        self.dwsl.append(dws32)

        dws41 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(256 * scale),
                num_filters1=256,
                num_filters2=256,
                num_groups=256,
                stride=1,
                scale=scale),
            name="conv4_1")
        self.dwsl.append(dws41)

        dws42 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(256 * scale),
                num_filters1=256,
                num_filters2=512,
                num_groups=256,
                stride=2,
                scale=scale),
            name="conv4_2")
        self.dwsl.append(dws42)

        for i in range(5):
            tmp = self.add_sublayer(
                sublayer=DepthwiseSeparable(
                    num_channels=int(512 * scale),
                    num_filters1=512,
                    num_filters2=512,
                    num_groups=512,
                    stride=1,
                    scale=scale),
                name="conv5_" + str(i + 1))
            self.dwsl.append(tmp)

        dws56 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(512 * scale),
                num_filters1=512,
                num_filters2=1024,
                num_groups=512,
                stride=2,
                scale=scale),
            name="conv5_6")
        self.dwsl.append(dws56)

        dws6 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(1024 * scale),
                num_filters1=1024,
                num_filters2=1024,
                num_groups=1024,
                stride=1,
                scale=scale),
            name="conv6")
        self.dwsl.append(dws6)

        self.pool2d_avg = Pool2D(pool_type='avg', global_pooling=True)

        self.out = Linear(
            int(1024 * scale),
            class_dim,
            param_attr=ParamAttr(
                initializer=MSRA(), name=self.full_name() + "fc7_weights"),
            bias_attr=ParamAttr(name=self.full_name() + "fc7_offset"))

    @to_static
    def forward(self, inputs):
        y = self.conv1(inputs)
        for dws in self.dwsl:
            y = dws(y)
        y = self.pool2d_avg(y)
        y = fluid.layers.reshape(y, shape=[-1, 1024])
        y = self.out(y)
        return y


class InvertedResidualUnit(fluid.dygraph.Layer):
    def __init__(
            self,
            num_channels,
            num_in_filter,
            num_filters,
            stride,
            filter_size,
            padding,
            expansion_factor, ):
        super(InvertedResidualUnit, self).__init__()
        num_expfilter = int(round(num_in_filter * expansion_factor))
        self._expand_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_expfilter,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            num_groups=1)

        self._bottleneck_conv = ConvBNLayer(
            num_channels=num_expfilter,
            num_filters=num_expfilter,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            num_groups=num_expfilter,
            act=None,
            use_cudnn=True)

        self._linear_conv = ConvBNLayer(
            num_channels=num_expfilter,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            num_groups=1)

    def forward(self, inputs, ifshortcut):
        y = self._expand_conv(inputs, if_act=True)
        y = self._bottleneck_conv(y, if_act=True)
        y = self._linear_conv(y, if_act=False)
        if ifshortcut:
            y = fluid.layers.elementwise_add(inputs, y)
        return y


class InvresiBlocks(fluid.dygraph.Layer):
    def __init__(self, in_c, t, c, n, s):
        super(InvresiBlocks, self).__init__()

        self._first_block = InvertedResidualUnit(
            num_channels=in_c,
            num_in_filter=in_c,
            num_filters=c,
            stride=s,
            filter_size=3,
            padding=1,
            expansion_factor=t)

        self._inv_blocks = []
        for i in range(1, n):
            tmp = self.add_sublayer(
                sublayer=InvertedResidualUnit(
                    num_channels=c,
                    num_in_filter=c,
                    num_filters=c,
                    stride=1,
                    filter_size=3,
                    padding=1,
                    expansion_factor=t),
                name=self.full_name() + "_" + str(i + 1))
            self._inv_blocks.append(tmp)

    def forward(self, inputs):
        y = self._first_block(inputs, ifshortcut=False)
        for inv_block in self._inv_blocks:
            y = inv_block(y, ifshortcut=True)
        return y


class MobileNetV2(fluid.dygraph.Layer):
    def __init__(self, class_dim=1000, scale=1.0):
        super(MobileNetV2, self).__init__()
        self.scale = scale
        self.class_dim = class_dim

        bottleneck_params_list = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        #1. conv1
        self._conv1 = ConvBNLayer(
            num_channels=3,
            num_filters=int(32 * scale),
            filter_size=3,
            stride=2,
            act=None,
            padding=1)

        #2. bottleneck sequences
        self._invl = []
        i = 1
        in_c = int(32 * scale)
        for layer_setting in bottleneck_params_list:
            t, c, n, s = layer_setting
            i += 1
            tmp = self.add_sublayer(
                sublayer=InvresiBlocks(
                    in_c=in_c, t=t, c=int(c * scale), n=n, s=s),
                name='conv' + str(i))
            self._invl.append(tmp)
            in_c = int(c * scale)

        #3. last_conv
        self._out_c = int(1280 * scale) if scale > 1.0 else 1280
        self._conv9 = ConvBNLayer(
            num_channels=in_c,
            num_filters=self._out_c,
            filter_size=1,
            stride=1,
            act=None,
            padding=0)

        #4. pool
        self._pool2d_avg = Pool2D(pool_type='avg', global_pooling=True)

        #5. fc
        tmp_param = ParamAttr(name=self.full_name() + "fc10_weights")
        self._fc = Linear(
            self._out_c,
            class_dim,
            param_attr=tmp_param,
            bias_attr=ParamAttr(name="fc10_offset"))

    @to_static
    def forward(self, inputs):
        y = self._conv1(inputs, if_act=True)
        for inv in self._invl:
            y = inv(y)
        y = self._conv9(y, if_act=True)
        y = self._pool2d_avg(y)
        y = fluid.layers.reshape(y, shape=[-1, self._out_c])
        y = self._fc(y)
        return y


def create_optimizer(args, parameter_list):
    optimizer = fluid.optimizer.Momentum(
        learning_rate=args.base_lr,
        momentum=args.momentum_rate,
        regularization=fluid.regularizer.L2Decay(args.l2_decay),
        parameter_list=parameter_list)

    return optimizer


def fake_data_reader(batch_size, label_size):
    local_random = np.random.RandomState(SEED)

    def reader():
        batch_data = []
        while True:
            img = local_random.random_sample([3, 224, 224]).astype('float32')
            label = local_random.randint(0, label_size, [1]).astype('int64')
            batch_data.append([img, label])
            if len(batch_data) == batch_size:
                yield batch_data
                batch_data = []

    return reader


def parse_args():
    parser = argparse.ArgumentParser("ResNet model benchmark.")
    parser.add_argument(
        '--model_type', type=str, default="v1", help='Model type, v1 or v2')
    # parser.add_argument(
    #     '--to_static', type=bool, default=True, help='whether to train model in static mode.')
    parser.add_argument(
        '--batch_size', type=int, default=64, help='The minibatch size.')
    parser.add_argument(
        '--pass_num', type=int, default=20, help='The number of passes.')
    parser.add_argument(
        '--class_num', type=int, default=102, help='The number of classes')
    parser.add_argument(
        '--base_lr',
        type=float,
        default=0.001,
        help='The basic learning rate.')
    parser.add_argument(
        '--momentum_rate',
        type=float,
        default=0.99,
        help='The momentum rate of optimizer.')
    parser.add_argument(
        '--l2_decay', type=float, default=1e-1, help='L2 decay value.')
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


def reader_decorator(reader):
    def __reader__():
        for item in reader():
            img = np.array(item[0]).astype('float32').reshape(3, 224, 224)
            label = np.array(item[1]).astype('int64').reshape(1)
            yield img, label

    return __reader__


def train(args, to_static):
    program_translator.enable(to_static)

    # set random seed
    np.random.seed(SEED)
    paddle.seed(SEED)
    paddle.framework.random._manual_program_seed(SEED)
    # set device
    device = 'gpu:0' if paddle.fluid.is_compiled_with_cuda(
    ) and args.device == 'GPU' else 'cpu'
    paddle.set_device(device)

    if "v1" in args.model_type:
        net = MobileNetV1(class_dim=args.class_num, scale=1.0)
    elif "v2" in args.model:
        net = MobileNetV2(class_dim=args.class_num, scale=1.0)
    else:
        print("wrong model name, please try model = v1 or v2")
        exit()

    optimizer = create_optimizer(args=args, parameter_list=net.parameters())

    # load flowers data
    train_reader = paddle.batch(
        reader_decorator(paddle.dataset.flowers.train(use_xmap=False)),
        batch_size=args.batch_size,
        drop_last=True)
    data_loader = paddle.io.DataLoader.from_generator(
        capacity=5, iterable=True)
    data_loader.set_sample_list_generator(train_reader)

    # 4. train loop
    net.train()
    args.pass_num = 1
    for pass_id in range(args.pass_num):
        total_loss = 0.0
        total_acc1 = 0.0
        total_acc5 = 0.0
        total_sample = 0
        cost_time = 0.

        for batch_id, data in enumerate(data_loader()):
            start_time = time.time()
            img, label = data
            out = net(img)

            softmax_out = fluid.layers.softmax(out, use_cudnn=False)
            loss = fluid.layers.cross_entropy(input=softmax_out, label=label)
            avg_loss = fluid.layers.mean(x=loss)
            acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
            acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

            # backward
            avg_loss.backward()
            t_end_back = time.time()
            optimizer.minimize(avg_loss)
            net.clear_gradients()

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

            if batch_id == 100:
                break

    return total_loss.numpy()


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
