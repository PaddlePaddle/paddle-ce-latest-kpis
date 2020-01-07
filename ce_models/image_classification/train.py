import os
import numpy as np
import time
import sys
import paddle
import paddle.fluid as fluid
from se_resnext import SE_ResNeXt
from mobilenet import mobile_net
import paddle.dataset.flowers as flowers
import reader

import argparse
import functools
import paddle.fluid.layers.ops as ops
from utility import add_arguments, print_arguments
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
import math

from continuous_evaluation import (train_acc_top1_kpi, train_acc_top5_kpi,
                                   train_cost_kpi, train_speed_kpi)
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',   int,  256, "Minibatch size.")
add_arg('num_layers',   int,  50,  "How many layers for SE-ResNeXt model.")
add_arg('parallel_exe', bool, True, "Whether to use ParallelExecutor to train or not.")
add_arg('init_model', str, None, "Whether to use initialized model.")
add_arg('pretrained_model', str, None, "Whether to use pretrained model.")
add_arg('lr_strategy', str, "cosine_decay",
        "Set the learning rate decay strategy.")
add_arg('model', str, "se_resnext", "Set the network to use.")


def cosine_decay(learning_rate, step_each_epoch, epochs=120):
    """Applies cosine decay to the learning rate.
    lr = 0.05 * (math.cos(epoch * (math.pi / 120)) + 1)
    """
    global_step = _decay_step_counter()

    epoch = ops.floor(global_step / step_each_epoch)
    decayed_lr = learning_rate * \
                 (ops.cos(epoch * (math.pi / epochs)) + 1)/2
    return decayed_lr


def train_parallel_exe(args,
                       learning_rate,
                       batch_size,
                       num_passes,
                       init_model=None,
                       pretrained_model=None,
                       model_save_dir='model',
                       parallel=True,
                       use_nccl=True,
                       lr_strategy=None,
                       layers=50):
    class_dim = 1000
    image_shape = [3, 224, 224]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    if args.model is 'se_resnext':
        out = SE_ResNeXt(input=image, class_dim=class_dim, layers=layers)
    else:
        out = mobile_net(img=image, class_dim=class_dim)

    cost = fluid.layers.cross_entropy(input=out, label=label)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
    avg_cost = fluid.layers.mean(x=cost)

    test_program = fluid.default_main_program().clone(for_test=True)

    if "piecewise_decay" in lr_strategy:
        bd = lr_strategy["piecewise_decay"]["bd"]
        lr = lr_strategy["piecewise_decay"]["lr"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
    elif "cosine_decay" in lr_strategy:
        step_each_epoch = lr_strategy["cosine_decay"]["step_each_epoch"]
        epochs = lr_strategy["cosine_decay"]["epochs"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=cosine_decay(
                learning_rate=learning_rate,
                step_each_epoch=step_each_epoch,
                epochs=epochs),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
    else:
        optimizer = fluid.optimizer.Momentum(
            learning_rate=learning_rate,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))

    opts = optimizer.minimize(avg_cost)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    fluid.default_startup_program.random_seed = 1000
    exe.run(fluid.default_startup_program())

    if init_model is not None:
        fluid.io.load_persistables(exe, init_model)

    if pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    train_reader = paddle.batch(flowers.train(), batch_size=batch_size)
    test_reader = paddle.batch(flowers.test(), batch_size=batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    train_exe = fluid.ParallelExecutor(use_cuda=True, loss_name=avg_cost.name)
    test_exe = fluid.ParallelExecutor(
        use_cuda=True, main_program=test_program, share_vars_from=train_exe)

    fetch_list = [avg_cost.name, acc_top1.name, acc_top5.name]
    train_speed = []
    for pass_id in range(num_passes):
        train_info = [[], [], []]
        test_info = [[], [], []]
        pass_time = 0
        pass_num = 0
        pass_speed = 0.0
        for batch_id, data in enumerate(train_reader()):
            t1 = time.time()
            loss, acc1, acc5 = train_exe.run(fetch_list,
                                             feed=feeder.feed(data))
            t2 = time.time()
            period = t2 - t1
            pass_time += period
            pass_num += len(data)
            loss = np.mean(np.array(loss))
            acc1 = np.mean(np.array(acc1))
            acc5 = np.mean(np.array(acc5))
            train_info[0].append(loss)
            train_info[1].append(acc1)
            train_info[2].append(acc5)
            if batch_id % 10 == 0:
                print("Pass {0}, trainbatch {1}, loss {2}, \
                       acc1 {3}, acc5 {4} time {5}"
                                                   .format(pass_id, \
                       batch_id, loss, acc1, acc5, \
                       "%2.2f sec" % period))
                sys.stdout.flush()

        train_loss = np.array(train_info[0]).mean()
        train_acc1 = np.array(train_info[1]).mean()
        train_acc5 = np.array(train_info[2]).mean()
        pass_speed = pass_num / pass_time
        train_speed.append(pass_speed)
        if pass_id == num_passes - 1:
            train_acc_top1_kpi.add_record(train_acc1)
            train_acc_top5_kpi.add_record(train_acc5)
            train_cost_kpi.add_record(train_loss)
            mean_pass_speed = np.array(pass_speed).mean()
            train_speed_kpi.add_record(mean_pass_speed)
        for data in test_reader():
            t1 = time.time()
            loss, acc1, acc5 = test_exe.run(fetch_list, feed=feeder.feed(data))
            t2 = time.time()
            period = t2 - t1
            loss = np.mean(np.array(loss))
            acc1 = np.mean(np.array(acc1))
            acc5 = np.mean(np.array(acc5))
            test_info[0].append(loss)
            test_info[1].append(acc1)
            test_info[2].append(acc5)
            if batch_id % 10 == 0:
                print("Pass {0},testbatch {1},loss {2}, \
                       acc1 {3},acc5 {4},time {5}"
                                                  .format(pass_id, \
                       batch_id, loss, acc1, acc5, \
                       "%2.2f sec" % period))
                sys.stdout.flush()

        test_loss = np.array(test_info[0]).mean()
        test_acc1 = np.array(test_info[1]).mean()
        test_acc5 = np.array(test_info[2]).mean()

        print("End pass {0}, train_loss {1}, train_acc1 {2}, train_acc5 {3}, \
               test_loss {4}, test_acc1 {5}, test_acc5 {6}, pass_time {7}, train_speed {8}"
                                                           .format(pass_id, \
              train_loss, train_acc1, train_acc5, test_loss, test_acc1, \
              test_acc5, pass_time, pass_num / pass_time))
        sys.stdout.flush()
    train_acc_top1_kpi.persist()
    train_acc_top5_kpi.persist()
    train_cost_kpi.persist()
    train_speed_kpi.persist()


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    total_images = 1281167
    batch_size = args.batch_size
    step = int(total_images / batch_size + 1)
    num_epochs = 5

    learning_rate_mode = args.lr_strategy
    lr_strategy = {}
    if learning_rate_mode == "piecewise_decay":
        epoch_points = [30, 60, 90]
        bd = [e * step for e in epoch_points]
        lr = [0.1, 0.01, 0.001, 0.0001]
        lr_strategy[learning_rate_mode] = {"bd": bd, "lr": lr}
    elif learning_rate_mode == "cosine_decay":
        lr_strategy[learning_rate_mode] = {
            "step_each_epoch": step,
            "epochs": num_epochs
        }
    else:
        lr_strategy = None

    use_nccl = True
    # layers: 50, 152
    layers = args.num_layers
    method = train_parallel_exe if args.parallel_exe else train_parallel_do
    init_model = args.init_model if args.init_model else None
    pretrained_model = args.pretrained_model if args.pretrained_model else None
    method(
        args,
        learning_rate=0.1,
        batch_size=batch_size,
        num_passes=num_epochs,
        init_model=init_model,
        pretrained_model=pretrained_model,
        parallel=True,
        use_nccl=True,
        lr_strategy=lr_strategy,
        layers=layers)
