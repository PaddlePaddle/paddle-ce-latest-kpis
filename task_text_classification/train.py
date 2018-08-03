import sys
import time
import unittest
import contextlib

import paddle.fluid as fluid
import paddle
import argparse
import utils
from nets import bow_net
from nets import cnn_net
from nets import lstm_net
from nets import gru_net
from continuous_evaluation import *
fluid.default_startup_program().random_seed = 99

def parse_args():
    parser = argparse.ArgumentParser("text_classification model benchmark.")
    parser.add_argument(
        '--model', type=str, default="lstm", help='model to run.')
    parser.add_argument(
        '--gpu_card_num', type=int, default=1, help='gpu card num used.')

    args = parser.parse_args()
    return args

def train(train_reader,
          word_dict,
          network,
          use_cuda,
          parallel,
          save_dirname,
          lr=0.2,
          batch_size=128,
          pass_num=30):
    """
    train network
    """
    args = parse_args()
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)

    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    if not parallel:
        cost, acc, prediction = network(data, label, len(word_dict))
    else:
        places = fluid.layers.device.get_places()
        pd = fluid.layers.ParallelDo(places)
        with pd.do():
            cost, acc, prediction = network(
                pd.read_input(data), pd.read_input(label), len(word_dict))

            pd.write_output(cost)
            pd.write_output(acc)

        cost, acc = pd()
        cost = fluid.layers.mean(cost)
        acc = fluid.layers.mean(acc)

    sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=lr)
    sgd_optimizer.minimize(cost)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[data, label], place=place)

    exe.run(fluid.default_startup_program())
    total_time = 0.0
    newest_avg_cost = 0.0
    for pass_id in xrange(pass_num):
        start_time = time.time()
        data_size, data_count, total_acc, total_cost = 0, 0, 0.0, 0.0
        for data in train_reader():
            avg_cost_np, avg_acc_np = exe.run(fluid.default_main_program(),
                                              feed=feeder.feed(data),
                                              fetch_list=[cost, acc])
            data_size = len(data)
            total_acc += data_size * avg_acc_np
            total_cost += data_size * avg_cost_np
            data_count += data_size
        avg_cost = total_cost / data_count
        newest_avg_cost = avg_cost
        t1 = time.time()
        total_time += t1 - start_time
        avg_acc = total_acc / data_count
        print("pass_id: %d, avg_acc: %f, avg_cost: %f" %
              (pass_id, avg_acc, avg_cost))
        if pass_id == pass_num - 1:
            if args.gpu_card_num == 1:
                lstm_train_cost_kpi.add_record(newest_avg_cost)
                lstm_pass_duration_kpi.add_record(total_time / pass_num)
            else:
                lstm_train_cost_kpi_card4.add_record(newest_avg_cost)
                lstm_pass_duration_kpi_card4.add_record(total_time / pass_num)

        epoch_model = save_dirname + "/" + "epoch" + str(pass_id)
        fluid.io.save_inference_model(epoch_model, ["words", "label"], acc,
                                      exe)
    if args.gpu_card_num == 1:
        lstm_train_cost_kpi.persist()
        lstm_pass_duration_kpi.persist()
    else:
        lstm_train_cost_kpi_card4.persist()
        lstm_pass_duration_kpi_card4.persist()

def train_net():
    args = parse_args()
    word_dict, train_reader, test_reader = utils.prepare_data(
        "imdb", self_dict=False, batch_size=128, buf_size=50000)

    if args.model == "bow":
        train(
            train_reader,
            word_dict,
            bow_net,
            use_cuda=False,
            parallel=False,
            save_dirname="bow_model",
            lr=0.002,
            pass_num=30,
            batch_size=128)
    elif args.model == "cnn":
        train(
            train_reader,
            word_dict,
            cnn_net,
            use_cuda=True,
            parallel=False,
            save_dirname="cnn_model",
            lr=0.01,
            pass_num=30,
            batch_size=4)
    elif args.model == "lstm":
        train(
            train_reader,
            word_dict,
            lstm_net,
            use_cuda=True,
            parallel=True,
            save_dirname="lstm_model",
            lr=0.05,
            pass_num=15,
            batch_size=4)
    elif args.model == "gru":
        train(
            train_reader,
            word_dict,
            lstm_net,
            use_cuda=True,
            parallel=False,
            save_dirname="gru_model",
            lr=0.05,
            pass_num=30,
            batch_size=128)
    else:
        print("network name cannot be found!")
        sys.exit(1)


if __name__ == "__main__":
    train_net()
