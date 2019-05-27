"""
VGG16 benchmark in Fluid
"""
from __future__ import print_function

import sys
import time
import numpy as np
import commands
import subprocess
import threading
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import argparse
import functools

from continuous_evaluation import tracking_kpis

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--batch_size', type=int, default=128, help="Batch size for training.")
parser.add_argument(
    '--skip_batch_num',
    type=int,
    default=5,
    help='The first num of minibatch num to skip, for better performance test')
parser.add_argument(
    '--iterations', type=int, default=80, help='The number of minibatches.')
parser.add_argument(
    '--learning_rate',
    type=float,
    default=1e-3,
    help="Learning rate for training.")
parser.add_argument('--pass_num', type=int, default=50, help="No. of passes.")
parser.add_argument(
    '--device',
    type=str,
    default='GPU',
    choices=['CPU', 'GPU'],
    help="The device type.")
parser.add_argument(
    "--gpu_id",
    type=int,
    default=3,
    help="The GPU Card Id. (default: %(default)d)")
parser.add_argument(
    '--data_format',
    type=str,
    default='NCHW',
    choices=['NCHW', 'NHWC'],
    help='The data order, now only support NCHW.')
parser.add_argument(
    '--data_set',
    type=str,
    default='cifar10',
    choices=['cifar10', 'flowers'],
    help='Optional dataset for benchmark.')
parser.add_argument(
    '--with_test',
    action='store_true',
    help='If set, test the testset during training.')
args = parser.parse_args()


def vgg16_bn_drop(input):
    """
    vgg16_bn_drop
    """

    def conv_block(input, num_filter, groups, dropouts):
        """
        conv_block
        """
        return fluid.nets.img_conv_group(
            input=input,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    return fc2


def main():
    """
    main
    """
    if args.data_set == "cifar10":
        classdim = 10
        if args.data_format == 'NCHW':
            data_shape = [3, 32, 32]
        else:
            data_shape = [32, 32, 3]
    else:
        classdim = 102
        if args.data_format == 'NCHW':
            data_shape = [3, 224, 224]
        else:
            data_shape = [224, 224, 3]

    # Input data
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # Train program
    net = vgg16_bn_drop(images)
    predict = fluid.layers.fc(input=net, size=classdim, act='softmax')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    # Evaluator
    batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
    batch_acc = fluid.layers.accuracy(
        input=predict, label=label, total=batch_size_tensor)

    # inference program
    inference_program = fluid.default_main_program().clone(for_test=True)

    # Optimization
    optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
    opts = optimizer.minimize(avg_cost)

    fluid.memory_optimize(fluid.default_main_program())

    # Initialize executor
    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(0)
    exe = fluid.Executor(place)

    # Parameter initialization
    exe.run(fluid.default_startup_program())

    # data reader
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.cifar.train10()
            if args.data_set == 'cifar10' else paddle.dataset.flowers.train(),
            buf_size=5120),
        batch_size=args.batch_size)
    test_reader = paddle.batch(
        paddle.dataset.cifar.test10()
        if args.data_set == 'cifar10' else paddle.dataset.flowers.test(),
        batch_size=args.batch_size)

    # test
    def test(exe):
        """
        test
        """
        test_accuracy = fluid.average.WeightedAverage()
        for batch_id, data in enumerate(test_reader()):
            img_data = np.array(
                map(lambda x: x[0].reshape(data_shape), data)).astype(
                    "float32")
            y_data = np.array(map(lambda x: x[1], data)).astype("int64")
            y_data = y_data.reshape([-1, 1])

            acc, weight = exe.run(inference_program,
                                  feed={"pixel": img_data,
                                        "label": y_data},
                                  fetch_list=[batch_acc, batch_size_tensor])
            test_accuracy.add(value=acc, weight=weight)
        return test_accuracy.eval()

    train_acc_kpi = None
    for kpi in tracking_kpis:
        if kpi.name == '%s_%s_train_acc' % (args.data_set, args.batch_size):
            train_acc_kpi = kpi
    train_speed_kpi = None
    for kpi in tracking_kpis:
        if kpi.name == '%s_%s_train_speed' % (args.data_set, args.batch_size):
            train_speed_kpi = kpi

    iters, num_samples, start_time = 0, 0, time.time()
    accuracy = fluid.average.WeightedAverage()
    for pass_id in range(args.pass_num):
        accuracy.reset()
        train_accs = []
        train_losses = []
        for batch_id, data in enumerate(train_reader()):
            if iters == args.skip_batch_num:
                start_time = time.time()
                num_samples = 0
            if iters == args.iterations:
                break
            img_data = np.array(
                map(lambda x: x[0].reshape(data_shape), data)).astype(
                    "float32")
            y_data = np.array(map(lambda x: x[1], data)).astype("int64")
            y_data = y_data.reshape([-1, 1])

            loss, acc, weight = exe.run(
                fluid.default_main_program(),
                feed={"pixel": img_data,
                      "label": y_data},
                fetch_list=[avg_cost, batch_acc, batch_size_tensor],
                use_program_cache=False)
            accuracy.add(value=acc, weight=weight)
            iters += 1
            num_samples += len(y_data)
            if (batch_id % 10) == 0:
                print(
                    "Pass = %d, Iter = %d, Loss = %f, Accuracy = %f" %
                    (pass_id, iters, loss, acc)
                )  # The accuracy is the accumulation of batches, but not the current batch.

        # pass_train_acc = accuracy.eval()
        train_losses.append(loss)
        train_accs.append(acc)
        print("Pass: %d, Loss: %f, Train Accuray: %f\n" %
              (pass_id, np.mean(train_losses), np.mean(train_accs)))
        train_elapsed = time.time() - start_time
        examples_per_sec = num_samples / train_elapsed
        print('\nTotal examples: %d, total time: %.5f, %.5f examples/sed\n' %
              (num_samples, train_elapsed, examples_per_sec))
        #train_acc_kpi.add_record(np.array(train_accs, dtype='float32'))
        train_speed_kpi.add_record(np.array(examples_per_sec, dtype='float32'))
        # evaluation
        if args.with_test:
            pass_test_acc = test(exe)
        break

#train_acc_kpi.persist()
    train_speed_kpi.persist()


def print_arguments():
    """
    print_arguments
    """
    print('----------- vgg Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


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


def save_gpu_data(mem_list):
    gpu_memory_kpi = None
    for kpi in tracking_kpis:
        if kpi.name == '%s_%s_gpu_memory' % (args.data_set, args.batch_size):
            gpu_memory_kpi = kpi
    gpu_memory_kpi.add_record(max(mem_list))
    gpu_memory_kpi.persist()


if __name__ == "__main__":
    print_arguments()
    global is_alive
    is_alive = True
    collect_memory_thread = threading.Thread(
        target=collect_gpu_memory_data, args=(is_alive, ))
    collect_memory_thread.setDaemon(True)
    collect_memory_thread.start()
    main()
    is_alive = False
