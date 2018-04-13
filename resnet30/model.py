from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time
import logging
import json

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

from continuous_evaluation import (train_cost_kpi, train_duration_kpi,
                                   tracking_kpis)

logger = logging.getLogger(__name__)


# model configs
def conv_bn_layer(input, ch_out, filter_size, stride, padding, act='relu'):
    conv1 = fluid.layers.conv2d(
        input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv1, act=act)


def shortcut(input, ch_out, stride):
    ch_in = input.shape[1]
    if ch_in != ch_out:
        return conv_bn_layer(input, ch_out, 1, stride, 0, None)
    else:
        return input


def basicblock(input, ch_out, stride):
    short = shortcut(input, ch_out, stride)
    conv1 = conv_bn_layer(input, ch_out, 3, stride, 1)
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1, act=None)
    return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')


def bottleneck(input, ch_out, stride):
    short = shortcut(input, ch_out * 4, stride)
    conv1 = conv_bn_layer(input, ch_out, 1, stride, 0)
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1)
    conv3 = conv_bn_layer(conv2, ch_out * 4, 1, 1, 0, act=None)
    return fluid.layers.elementwise_add(x=short, y=conv3, act='relu')


def layer_warp(block_func, input, ch_out, count, stride):
    res_out = block_func(input, ch_out, stride)
    for i in range(1, count):
        res_out = block_func(res_out, ch_out, 1)
    return res_out


def resnet_cifar10(input, class_dim, depth=32):
    assert (depth - 2) % 6 == 0

    n = (depth - 2) // 6

    conv1 = conv_bn_layer(
        input=input, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 64, n, 2)
    pool = fluid.layers.pool2d(
        input=res3, pool_size=8, pool_type='avg', pool_stride=1)
    out = fluid.layers.fc(input=pool, size=class_dim, act='softmax')
    return out


def train(batch_size, device, pass_num, iterations):
    print('iterations', iterations)
    class_dim = 10
    # NCHW
    dshape = [3, 32, 32]
    input = fluid.layers.data(name='data', shape=dshape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    predict = resnet_cifar10(input, class_dim)
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    optimizer = fluid.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
    opts = optimizer.minimize(avg_cost)
    # accuracy = fluid.evaluator.Evaluator(input=predict, label=label)

    # inference program
    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        # test_target = accuracy.metrics + accuracy.states
        test_target = [predict, avg_cost]
        inference_program = fluid.io.get_inference_program(test_target)

    fluid.memory_optimize(fluid.default_main_program())

    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.cifar.train10(), buf_size=5120),
        batch_size=batch_size)

    test_reader = paddle.batch(
        paddle.dataset.cifar.test10(), batch_size=batch_size)

    def test(exe):
        # accuracy.reset(exe)
        for batch_id, data in enumerate(test_reader()):
            img_data = np.array(map(lambda x: x[0].reshape(dshape),
                                    data)).astype("float32")
            y_data = np.array(map(lambda x: x[1], data)).astype("int64")
            y_data = y_data.reshape([-1, 1])

            # print('image_data', img_data)
            # print('y_data', y_data)

            predict_, avg_cost_ = exe.run(
                inference_program,
                feed={
                    "data": img_data,
                    "label": y_data
                },
                fetch_list=[predict, avg_cost])
            return avg_cost

        # return accuracy.eval(exe)

    place = core.CPUPlace() if device == 'CPU' else core.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    for pass_id in range(1):
        logger.warning('Pass {}'.format(pass_id))
        # accuracy.reset(exe)
        iter = 0
        for batch_id, data in enumerate(train_reader()):
            logger.warning('Batch {}'.format(batch_id))
            batch_start = time.time()
            if iter == iterations:
                break
            image = np.array(map(lambda x: x[0].reshape(dshape),
                                 data)).astype('float32')
            label = np.array(map(lambda x: x[1], data)).astype('int64')
            label = label.reshape([-1, 1])
            avg_cost_ = exe.run(
                fluid.default_main_program(),
                feed={
                    'data': image,
                    'label': label
                },
                fetch_list=[avg_cost])
            batch_end = time.time()
            print('avg_cost', np.array(avg_cost_, dtype='float32'))
            train_cost_kpi.add_record(np.array(avg_cost_, dtype='float32'))
            train_duration_kpi.add_record(batch_end - batch_start)

            iter += 1

            # test_start = time.time()
            # test(exe)
            # test_end = time.time()
            # valid_tracker.add(test_end - test_start, pass_test_acc)


def parse_args():
    parser = argparse.ArgumentParser('model')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--device', type=str, choices=('CPU', 'GPU'))
    parser.add_argument('--iters', type=int)
    args = parser.parse_args()
    return args


args = parse_args()
train(args.batch_size, args.device, 1, args.iters)

for kpi in tracking_kpis:
    kpi.persist()
