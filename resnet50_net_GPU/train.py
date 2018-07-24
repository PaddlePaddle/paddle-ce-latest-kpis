from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import functools
import numpy as np
import time
import os
import sys
import commands
import subprocess
import threading

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.profiler as profiler

sys.path.append("../resnet50_net")
import models
import models.resnet

from continuous_evaluation import tracking_kpis

fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


def parse_args():
    parser = argparse.ArgumentParser('Convolution model benchmark.')
    parser.add_argument(
        '--model',
        type=str,
        choices=['resnet_imagenet', 'resnet_cifar10'],
        default='resnet_imagenet',
        help='The model architecture.')
    parser.add_argument(
        '--batch_size', type=int, default=32, help='The minibatch size.')
    parser.add_argument(
        '--skip_batch_num',
        type=int,
        default=5,
        help='The first num of minibatch num to skip, for better performance test'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=80,
        help='The number of minibatches.')
    parser.add_argument(
        '--pass_num', type=int, default=100, help='The number of passes.')
    parser.add_argument(
        '--data_format',
        type=str,
        default='NCHW',
        choices=['NCHW', 'NHWC'],
        help='The data data_format, now only support NCHW.')
    parser.add_argument(
        '--use_gpu',
        type=distutils.util.strtobool,
        default=False,
        help='Whether to use gpu.')
    parser.add_argument(
        '--reduce_strategy',
        type=str,
        default='AllReduce',
        choices=['AllReduce', 'Reduce', 'None'],
        help='The reduce strategy.')
    parser.add_argument(
        "--gpu_id",
        type=str,
        default=0,
        help="The GPU Cards Id. (default: %(default)d)")
    parser.add_argument(
        '--data_set',
        type=str,
        default='flowers',
        choices=['cifar10', 'flowers'],
        help='Optional dataset for benchmark.')

    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def get_cards(args):
    if args.use_gpu:
        cards = os.environ.get('CUDA_VISIBLE_DEVICES')
        cards = str(len(cards.split(",")))
    else:
        cards = os.environ.get('CPU_NUM')
    return cards


def record_kpi(pass_id, iter, pass_train_acc, total_train_time, im_num):
    # Record KPI
    cards = get_cards(args)

    if int(cards) > 1:
        run_info = args.reduce_strategy + "_" \
                + ("GPU" if args.use_gpu else "CPU") + "_" \
                + cards + "_Cards"
    else:
        run_info = ("GPU" if args.use_gpu else "CPU") + "_" \
                + cards + "_Cards"

    kpi_name_base = '%s_%s_%s_train' % (args.data_set, args.batch_size,
                                        run_info)

    train_acc_kpi = None
    for kpi in tracking_kpis:
        if kpi.name == kpi_name_base + "_acc":
            train_acc_kpi = kpi
            break
    assert train_acc_kpi is not None, kpi_name_base + "_acc" + " is not found."

    train_speed_kpi = None
    for kpi in tracking_kpis:
        if kpi.name == kpi_name_base + "_speed":
            train_speed_kpi = kpi
            break
    assert train_speed_kpi is not None, kpi_name_base + "_speed" + " is not found."

    # Record KPI
    if pass_id == args.pass_num - 1:
        train_acc_kpi.add_record(np.array(pass_train_acc, dtype='float32'))
        train_acc_kpi.persist()
    if total_train_time > 0.0 and iter != args.skip_batch_num:
        examples_per_sec = im_num / total_train_time
        sec_per_batch = total_train_time / \
            (iter * args.pass_num - args.skip_batch_num)
        train_speed_kpi.add_record(np.array(examples_per_sec, dtype='float32'))
        train_speed_kpi.persist()


def init_reader(args):
    train_reader = paddle.batch(
        paddle.dataset.cifar.train10()
        if args.data_set == 'cifar10' else paddle.dataset.flowers.train(),
        batch_size=args.batch_size)

    test_reader = paddle.batch(
        paddle.dataset.cifar.test10()
        if args.data_set == 'cifar10' else paddle.dataset.flowers.test(),
        batch_size=args.batch_size)
    return train_reader, test_reader


def get_parallel_executor(args, avg_cost, train_program, test_program):
    # Init Parameter
    place = core.CUDAPlace(0) if args.use_gpu else core.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.allow_op_delay = True
    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce \
        if args.reduce_strategy == "Reduce" \
        else fluid.BuildStrategy.ReduceStrategy.AllReduce

    train_exe = fluid.ParallelExecutor(
        loss_name=avg_cost.name,
        main_program=train_program,
        use_cuda=True if args.use_gpu else False,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    test_exe = fluid.ParallelExecutor(
        main_program=test_program,
        use_cuda=True if args.use_gpu else False,
        share_vars_from=train_exe,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)
    return train_exe, test_exe


def get_data_shape(args):
    if args.data_set == "cifar10":
        class_dim = 10
        if args.data_format == 'NCHW':
            dshape = [3, 32, 32]
        else:
            dshape = [32, 32, 3]
    else:
        class_dim = 102
        if args.data_format == 'NCHW':
            dshape = [3, 224, 224]
        else:
            dshape = [224, 224, 3]
    return dshape, class_dim


def run_benchmark(model, args):

    dshape, class_dim = get_data_shape(args)

    input = fluid.layers.data(name='data', shape=dshape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    predict = model(input, class_dim)
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    fluid.default_main_program().seed = 1
    fluid.default_startup_program().seed = 1

    batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
    batch_acc = fluid.layers.accuracy(
        input=predict, label=label, total=batch_size_tensor)

    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        inference_program = fluid.io.get_inference_program(
            target_vars=[batch_acc, batch_size_tensor])

    optimizer = fluid.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
    opts = optimizer.minimize(avg_cost)

    fluid.memory_optimize(fluid.default_main_program())

    # Init ParallelExecutor
    train_exe, test_exe = get_parallel_executor(args, avg_cost,
                                                fluid.default_main_program(),
                                                inference_program)

    # Prepare reader
    train_reader, test_reader = init_reader(args)

    def test(test_exe):
        test_accuracy = fluid.average.WeightedAverage()
        for batch_id, data in enumerate(test_reader()):
            if batch_id == args.iterations:
                break
            img_data = np.array(map(lambda x: x[0].reshape(dshape),
                                    data)).astype("float32")
            y_data = np.array(map(lambda x: x[1], data)).astype(
                "int64").reshape([-1, 1])

            acc, weight = test_exe.run(
                fetch_list=[batch_acc.name, batch_size_tensor.name],
                feed={"data": img_data,
                      "label": y_data})
            acc = float((acc * weight).sum() / weight.sum())
            weight = int(weight.sum())
            test_accuracy.add(value=acc, weight=weight)

        return test_accuracy.eval()

    im_num, total_train_time, total_iters = 0, 0.0, 0
    accuracy = fluid.average.WeightedAverage()
    fetch_list = [avg_cost.name, batch_acc.name, batch_size_tensor.name]

    for pass_id in range(args.pass_num):
        every_pass_loss = []
        accuracy.reset()
        iter, pass_duration = 0, 0.0
        for batch_id, data in enumerate(train_reader()):
            batch_start = time.time()
            if iter == args.iterations:
                break

            image = np.array(map(lambda x: x[0].reshape(dshape), data)).astype(
                'float32')
            label = np.array(map(lambda x: x[1], data)).astype(
                'int64').reshape([-1, 1])

            loss, acc, weight = train_exe.run(
                fetch_list=fetch_list, feed={'data': image,
                                             'label': label})

            acc = float((acc * weight).sum() / weight.sum())
            loss = (loss * weight).sum() / weight.sum()
            weight = int(weight.sum())
            accuracy.add(value=acc, weight=weight)

            if iter >= args.skip_batch_num or pass_id != 0:
                batch_duration = time.time() - batch_start
                pass_duration += batch_duration
                im_num += label.shape[0]

            every_pass_loss.append(loss)
            # print("Pass: %d, Iter: %d, loss: %s, acc: %s" %
            #      (pass_id, iter, str(loss), str(acc)))
            iter += 1
            total_iters += 1

        total_train_time += pass_duration
        pass_train_acc = accuracy.eval()
        pass_test_acc = test(test_exe)
        print(
            "Pass:%d, Loss:%f, Train Accuray:%f, Test Accuray:%f, Handle Images Duration: %f\n"
            % (pass_id, np.mean(every_pass_loss), pass_train_acc,
               pass_test_acc, pass_duration))

    record_kpi(pass_id, iter, pass_train_acc, total_train_time, im_num)

    examples_per_sec = im_num / total_train_time
    sec_per_batch = total_train_time / \
            (iter * args.pass_num - args.skip_batch_num)

    print('\nTotal examples: %d, total time: %.5f' %
          (im_num, total_train_time))
    print('%.5f examples/sec, %.5f sec/batch \n' %
          (examples_per_sec, sec_per_batch))


def collect_gpu_memory_data(alive):
    """
    collect the GPU memory data
    """
    global is_alive
    status, output = commands.getstatusoutput('rm -rf memory.txt')
    if status == 0:
        print('del memory.txt')
    command = "nvidia-smi --id=%s --query-compute-apps=used_memory --format=csv -lms 1 > memory.txt" % args.gpu_id
    p = subprocess.Popen(command, shell=True)
    if p.pid < 0:
        print('Get GPU memory data error')
    while (is_alive):
        time.sleep(1)
    p.kill()


if __name__ == '__main__':
    model_map = {
        'resnet_imagenet': models.resnet.resnet_imagenet,
        'resnet_cifar10': models.resnet.resnet_cifar10
    }

    args = parse_args()

    args.data_set = "cifar10" \
        if args.model == "resnet_cifar10" else "flowers"

    cards = get_cards(args)
    if int(cards) == 1:
        args.reduce_strategy = "None"

    print_arguments(args)

    global is_alive
    is_alive = True

    if args.data_format == 'NHWC':
        raise ValueError('Only support NCHW data_format now.')

    if args.use_gpu:
        collect_memory_thread = threading.Thread(
            target=collect_gpu_memory_data, args=(is_alive, ))
        collect_memory_thread.setDaemon(True)
        collect_memory_thread.start()

    run_benchmark(model_map[args.model], args)

    is_alive = False
