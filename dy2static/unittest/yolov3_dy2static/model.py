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

import time
import argparse
import os
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
from paddle.jit import ProgramTranslator

from yolov3 import YOLOv3
import dist_utils
import reader

SEED = 2020
PRINT_STEP = 1


def parse_args():
    parser = argparse.ArgumentParser("yolov3 model benchmark.")
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


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self):
        self.loss_sum = 0.0
        self.iter_cnt = 0

    def add_value(self, value):
        self.loss_sum += np.mean(value)
        self.iter_cnt += 1

    def get_mean_value(self):
        return self.loss_sum / self.iter_cnt


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value


def get_config(args):
    cfg = AttrDict()
    cfg.snapshot_iter = 2000
    # min valid area for gt boxes
    cfg.gt_min_area = -1
    # max target box number in an image
    cfg.max_box_num = 50
    # valid score threshold to include boxes
    cfg.valid_thresh = 0.005
    # threshold vale for box non-max suppression
    cfg.nms_thresh = 0.45
    # the number of top k boxes to perform nms
    cfg.nms_topk = 400
    # the number of output boxes after nms
    cfg.nms_posk = 100
    # score threshold for draw box in debug mode
    cfg.draw_thresh = 0.5
    # Use label smooth in class label
    cfg.label_smooth = True
    #
    # Model options
    #
    # input size
    cfg.input_size = 608
    # pixel mean values
    cfg.pixel_means = [0.485, 0.456, 0.406]
    # pixel std values
    cfg.pixel_stds = [0.229, 0.224, 0.225]
    # anchors box weight and height
    cfg.anchors = [
        10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198,
        373, 326
    ]
    # anchor mask of each yolo layer
    cfg.anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # IoU threshold to ignore objectness loss of pred box
    cfg.ignore_thresh = .7
    #
    # SOLVER options
    #
    # batch size
    cfg.batch_size = args.batch_size
    # derived learning rate the to get the final learning rate.
    cfg.learning_rate = 0.001
    # support both CPU and GPU
    cfg.use_gpu = fluid.is_compiled_with_cuda() and args.device == 'GPU'
    cfg.use_data_parallel = cfg.use_gpu
    cfg.use_multiprocess_reader = False
    # maximum number of iterations
    cfg.max_iter = 20 if cfg.use_gpu else 2
    # Disable mixup in last N iter
    cfg.no_mixup_iter = 10 if cfg.use_gpu else 1
    # warm up to learning rate
    cfg.warm_up_iter = 10 if cfg.use_gpu else 1
    cfg.warm_up_factor = 0.
    # lr steps_with_decay
    cfg.lr_steps = [400000, 450000]
    cfg.lr_gamma = 0.1
    # L2 regularization hyperparameter
    cfg.weight_decay = 0.0005
    # momentum with SGD
    cfg.momentum = 0.9
    #
    # ENV options
    #
    # Class number
    cfg.class_num = 80
    # dataset path
    cfg.train_file_list = 'annotations/instances_train2017.json'
    cfg.train_data_dir = 'train2017'
    cfg.val_file_list = 'annotations/instances_val2017.json'
    cfg.val_data_dir = 'val2017'
    cfg.dataset = ['coco2014']
    cfg.data_dir = os.path.join(
        os.path.expanduser("~"), ".cache/paddle/dataset/coco")
    return cfg


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
    cfg = get_config(args)

    # create model
    ch_in = 3
    yolov3_model = YOLOv3(cfg, ch_in)

    boundaries = cfg.lr_steps
    gamma = cfg.lr_gamma
    step_num = len(cfg.lr_steps)
    learning_rate = cfg.learning_rate
    values = [learning_rate * (gamma**i) for i in range(step_num + 1)]

    lr = fluid.dygraph.PiecewiseDecay(
        boundaries=boundaries, values=values, begin=0)

    lr = fluid.layers.linear_lr_warmup(
        learning_rate=lr,
        warmup_steps=cfg.warm_up_iter,
        start_lr=0.0,
        end_lr=cfg.learning_rate, )

    optimizer = fluid.optimizer.Momentum(
        learning_rate=lr,
        regularization=fluid.regularizer.L2Decay(cfg.weight_decay),
        momentum=cfg.momentum,
        parameter_list=yolov3_model.parameters())

    total_sample = 0

    input_size = cfg.input_size
    shuffle = True
    shuffle_seed = None
    total_iter = cfg.max_iter
    mixup_iter = total_iter - cfg.no_mixup_iter

    devices_num = 1
    random_sizes = [cfg.input_size]
    #'''
    for pass_id in range(args.pass_num):
        snapshot_loss = 0
        snapshot_time = 0
        train_reader = reader.train(
            input_size,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            total_iter=total_iter * devices_num,
            mixup_iter=mixup_iter * devices_num,
            random_sizes=random_sizes,
            use_multiprocess_reader=cfg.use_multiprocess_reader,
            use_gpu=cfg.use_gpu,
            cfg=cfg)
        #'''
        if cfg.use_data_parallel:
            train_reader = fluid.contrib.reader.distributed_batch_reader(
                train_reader)
        smoothed_loss = SmoothedValue()

        start_time = time.time()

        for iter_id, data in enumerate(train_reader()):
            #print ( 'iter_id = %d' % iter_id )
            prev_start_time = start_time
            start_time = time.time()
            img = np.array([x[0] for x in data]).astype('float32')
            img = to_variable(img)

            gt_box = np.array([x[1] for x in data]).astype('float32')
            gt_box = to_variable(gt_box)

            gt_label = np.array([x[2] for x in data]).astype('int32')
            gt_label = to_variable(gt_label)

            gt_score = np.array([x[3] for x in data]).astype('float32')
            gt_score = to_variable(gt_score)

            loss = yolov3_model(img, gt_box, gt_label, gt_score, None, None)
            smoothed_loss.add_value(np.mean(loss.numpy()))
            snapshot_loss += loss.numpy()
            snapshot_time += (start_time - prev_start_time) * 1000  # ms
            total_sample += 1

            if iter_id % PRINT_STEP == 0:
                print(
                    "To_static = {:s}, pass = {:d}, Iter = {:d}, Loss = {:.6f}, Elapse(ms) = {:.5f}"
                    .format(
                        str(to_static), pass_id, iter_id,
                        smoothed_loss.get_mean_value(), start_time -
                        prev_start_time))
            loss.backward()

            optimizer.minimize(loss)
            yolov3_model.clear_gradients()
        print(
            "To_static = {:s}, pass = {:d}, Loss = {:.6f}, Elapse(ms) = {:.5f}"
            .format(
                str(to_static), pass_id,
                smoothed_loss.get_mean_value(), snapshot_time))
    print('-----------------------------end-----------------------------')


def run_benchmark(args):
    # train in dygraph mode
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
    print('All done')
