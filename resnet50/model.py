from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import numpy as np
import time
import commands
import subprocess
import threading

import cProfile
import pstats
import StringIO

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.profiler as profiler

from continuous_evaluation import tracking_kpis

fluid.default_startup_program().random_seed = 91


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
        '--log_dir',
        '-f',
        type=str,
        default='./',
        help='The path of the log file')
    parser.add_argument(
        '--use_fake_data',
        action='store_true',
        help='use real data or fake data')
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
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    parser.add_argument(
        "--gpu_id",
        type=str,
        default='0,1,2,3',
        help="The GPU Card Id. (default: %(default)d)")
    parser.add_argument(
        '--data_set',
        type=str,
        default='flowers',
        choices=['cifar10', 'flowers'],
        help='Optional dataset for benchmark.')
    parser.add_argument(
        '--infer_only', action='store_true', help='If set, run forward only.')
    parser.add_argument(
        '--use_cprof', action='store_true', help='If set, use cProfile.')
    parser.add_argument(
        '--use_nvprof',
        action='store_true',
        help='If set, use nvprof for CUDA.')
    args = parser.parse_args()
    return args


def print_arguments(args):
    vars(args)['use_nvprof'] = (vars(args)['use_nvprof'] and
                                vars(args)['device'] == 'GPU')
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


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
    ch_in = input.shape[1] if args.data_format == 'NCHW' else input.shape[-1]
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


def resnet_imagenet(input, class_dim, depth=50, data_format='NCHW'):

    cfg = {
        18: ([2, 2, 2, 1], basicblock),
        34: ([3, 4, 6, 3], basicblock),
        50: ([3, 4, 6, 3], bottleneck),
        101: ([3, 4, 23, 3], bottleneck),
        152: ([3, 8, 36, 3], bottleneck)
    }
    stages, block_func = cfg[depth]
    conv1 = conv_bn_layer(input, ch_out=64, filter_size=7, stride=2, padding=3)
    pool1 = fluid.layers.pool2d(
        input=conv1, pool_type='avg', pool_size=3, pool_stride=2)
    res1 = layer_warp(block_func, pool1, 64, stages[0], 1)
    res2 = layer_warp(block_func, res1, 128, stages[1], 2)
    res3 = layer_warp(block_func, res2, 256, stages[2], 2)
    res4 = layer_warp(block_func, res3, 512, stages[3], 2)
    pool2 = fluid.layers.pool2d(
        input=res4,
        pool_size=7,
        pool_type='avg',
        pool_stride=1,
        global_pooling=True)
    out = fluid.layers.fc(input=pool2, size=class_dim, act='softmax')
    return out


def resnet_cifar10(input, class_dim, depth=32, data_format='NCHW'):
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


def run_benchmark(model, args):
    if args.use_cprof:
        pr = cProfile.Profile()
        pr.enable()

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

    # Input data
    image = fluid.layers.data(name='image', shape=dshape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    #Train program
    predict = model(image, class_dim)
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    # Evaluator
    batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
    batch_acc = fluid.layers.accuracy(
        input=predict, label=label, total=batch_size_tensor)

    # Optimization to minimize lost
    optimizer = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
    opts = optimizer.minimize(avg_cost)
    fluid.memory_optimize(fluid.default_main_program())

    # Initialize executor
    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    
   # Reader 
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
    train_reader = \
         paddle.batch(
            paddle.dataset.cifar.train10()
            if args.data_set == 'cifar10' else paddle.dataset.flowers.train(),
               batch_size=args.batch_size)
      

    test_reader = paddle.batch(
        paddle.dataset.cifar.test10()
        if args.data_set == 'cifar10' else paddle.dataset.flowers.test(),
        batch_size=args.batch_size)
    
    # Register test program
    test_program = fluid.default_main_program().clone(for_test=True)
    with fluid.program_guard(test_program):
        test_program = fluid.io.get_inference_program(
            target_vars=[batch_acc])

    # Define parallel exe
    train_exe = fluid.ParallelExecutor(use_cuda=True, loss_name=avg_cost.name)
    def test(exe):
        test_accuracy = []
        for batch_id, data in enumerate(test_reader()):

            acc, = exe.run(test_program,
                                  fetch_list=[batch_acc],
                                  feed=feeder.feed(data)
                                  )
            acc_avg = np.mean(np.array(acc))
            test_accuracy.append(acc_avg)

        return np.array(test_accuracy).mean()


    if args.use_fake_data:
        data = train_reader().next()
        image = np.array(map(lambda x: x[0].reshape(dshape), data)).astype(
            'float32')
        label = np.array(map(lambda x: x[1], data)).astype('int64')
        label = label.reshape([-1, 1])

    im_num = 0
    total_train_time = 0.0
    
    fetch_list=[avg_cost.name, batch_acc.name]
    #fetch_list=[avg_cost.name]
    for pass_id in range(args.pass_num):
        every_pass_loss = []
        every_pass_acc = []
        iter = 0
        pass_duration = 0.0
        for batch_id, data in enumerate(train_reader()):
            if iter == args.iterations:
                break
            batch_start = time.time()
            loss, acc = train_exe.run(
                fetch_list=fetch_list,
                feed=feeder.feed(data)
             )

            if iter >= args.skip_batch_num or pass_id != 0:
                batch_duration = time.time() - batch_start
                pass_duration += batch_duration
                im_num += len(data)
            loss_avg, acc_avg = np.mean(np.array(loss)), np.mean(np.array(acc))
            print("Pass: %d, Iter: %d, loss: %s, acc: %s" % \
                        (pass_id, batch_id, loss_avg, acc_avg))
            
            every_pass_loss.append(loss_avg)
            every_pass_acc.append(acc_avg)
            iter += 1

        total_train_time += pass_duration
        # Begin test
        pass_test_acc = test(exe)
        print(
            "Pass:%d, Loss:%f, Train Accuray:%f, Test Accuray:%f,\
                     Handle Images Duration: %f\n"
            % (pass_id, np.mean(every_pass_loss), np.mean(every_pass_acc),
               pass_test_acc, pass_duration))

    examples_per_sec = 0
    if total_train_time > 0.0 and iter != args.skip_batch_num:
        examples_per_sec = im_num / total_train_time
        sec_per_batch = total_train_time / \
            (iter * args.pass_num - args.skip_batch_num)
        print('\nTotal examples: %d, total time: %.5f' %
              (im_num, total_train_time))
        print('%.5f examples/sec, %.5f sec/batch \n' %
              (examples_per_sec, sec_per_batch))

    gpu_nums = len(args.gpu_id.split(','))
    save_kpi_data(gpu_nums, examples_per_sec, every_pass_acc)

    if args.device == 'GPU':
        # persist mem kpi
        save_gpu_data(gpu_nums)

    if args.use_cprof:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())


def save_kpi_data(gpu_nums, examples_per_sec, every_pass_acc):
    train_acc_kpi = None
    train_speed_kpi = None

    for kpi in tracking_kpis:
        if kpi.name == '%s_train_acc' % (args.data_set):
            train_acc_kpi = kpi

    for kpi in tracking_kpis:
        if kpi.name == '%s_train_speed' % (args.data_set):
            train_speed_kpi = kpi

    train_speed_kpi.add_record(np.array(examples_per_sec, dtype='float32'))
    train_speed_kpi.persist()

    if args.data_set == 'cifar10':
        train_acc_kpi.add_record(np.array(np.mean(every_pass_acc),\
                     dtype='float32'))
        train_acc_kpi.persist()


def collect_gpu_memory_data(alive):
    """
    collect the GPU memory data
    """
    global is_alive
    status, output = commands.getstatusoutput('rm -rf memory_*')
    if status == 0:
        print('del memory')
    pid_list = []
    for gpu_id in args.gpu_id.split(','):
        command = "nvidia-smi --id=%s --query-compute-apps=used_memory --format=csv\
                    -lms 1000 > memory_%s.txt" % (gpu_id, gpu_id)
        p = subprocess.Popen(command, shell=True)
        if p.pid < 0:
            print('Get GPU memory data error')
        else:
            pid_list.append(p)
           
    while (is_alive):
        time.sleep(1)
    for p in pid_list:
        p.kill()


def save_gpu_data(gpu_nums):
    mem_list = []
    status, output = commands.getstatusoutput('''cat memory*|\
             awk {'print $1'}|awk '{sum+=$1} END {print "Average = ", sum/NR}'\
              | awk {'print $3'} ''')
    mem = output.strip()
    gpu_memory_factor = None
    for kpi in tracking_kpis:
        if kpi.name == '%s_gpu_memory' % (args.data_set):
            gpu_memory_kpi = kpi
        if kpi.name == '%s_gpu_memory_card%s' % (args.data_set, gpu_nums):
            var['gpu_memory' + str(gpu_nums)] = kpi
    gpu_memory_kpi.add_record(np.array(mem, dtype='float32'))
    gpu_memory_kpi.persist()


if __name__ == '__main__':
    model_map = {
        'resnet_imagenet': resnet_imagenet,
        'resnet_cifar10': resnet_cifar10
    }
    args = parse_args()
    print_arguments(args)
    global is_alive
    is_alive = True
    if args.data_format == 'NHWC':
        raise ValueError('Only support NCHW data_format now.')
    if args.device == 'GPU':
        collect_memory_thread = threading.Thread(
            target=collect_gpu_memory_data, args=(is_alive, ))
        collect_memory_thread.setDaemon(True)
        collect_memory_thread.start()
    if args.use_nvprof and args.device == 'GPU':
        with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
            run_benchmark(model_map[args.model], args)
    else:
        run_benchmark(model_map[args.model], args)
        is_alive = False

