#!/bin/env python
# -*- encoding:utf-8 -*-
"""
fit a line demo
"""
import os
import paddle.v2 as paddle
import numpy as np

cluster_train_dir = "./train_data"
cluster_test_dir = "./test_data"

def cluster_data_reader(file_dir):
    """
    cluster data reader
    """
    def data_reader():
        """
        data reader
        """
        files = os.listdir(file_dir)
        for fi in files:
            with open(file_dir + '/' + fi, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ins = np.fromstring(line, dtype=float, sep=" ")
                        ins_data = ins[:-1]
                        label = ins[-1:]
                        yield ins_data, label
    return data_reader


def train():
    """
    model train
    """
    # init
    use_gpu_flag = os.getenv("PADDLE_USE_GPU", "0")
    print("use_gpu_flag", use_gpu_flag)
    trainer_id_str = os.getenv("PADDLE_TRAINER_ID", "0")
    paddle.init(use_gpu = use_gpu_flag,
            trainer_count = int(os.getenv("PADDLE_TRAINER_COUNT", "1")),
            port = int(os.getenv("PADDLE_PORT", "7164")),
            ports_num = int(os.getenv("PADDLE_PORTS_NUM", "1")),
            ports_num_for_sparse = int(os.getenv("PADDLE_PORTS_NUM_FOR_SPARSE", "1")),
            num_gradient_servers = int(os.getenv("PADDLE_NUM_GRADIENT_SERVERS", "1")),
            trainer_id = int(trainer_id_str),
            pservers = os.getenv("PADDLE_PSERVERS", "127.0.0.1"))
    # network config
    x = paddle.layer.data(name = 'x', type = paddle.data_type.dense_vector(13))
    y_predict = paddle.layer.fc(input = x, size = 1, act = paddle.activation.Linear())
    y = paddle.layer.data(name = 'y', type = paddle.data_type.dense_vector(1))
    cost = paddle.layer.square_error_cost(input = y_predict, label = y)

    # Save the inference topology to protobuf.
    inference_topology = paddle.topology.Topology(layers = y_predict)
    with open("./output/inference_topology.pkl", 'wb') as f:
        inference_topology.serialize_for_inference(f)

    # create parameters
    parameters = paddle.parameters.create(cost)

    # create optimizer
    optimizer = paddle.optimizer.Momentum(momentum = 0)

    is_local = os.getenv("PADDLE_IS_LOCAL", "0") == "1"
    trainer = paddle.trainer.SGD(
        cost = cost,
        parameters = parameters,
        update_equation = optimizer,
        is_local = is_local)

    feeding = {'x': 0, 'y': 1}

    #create reader
    train_reader = paddle.batch(
            paddle.reader.shuffle(
                cluster_data_reader(cluster_train_dir), buf_size = 500),
                batch_size = 2)
    # event_handler to print training and testing info
    def event_handler(event):
        """
        handle paddle event
        """
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f" % (
                    event.pass_id, event.batch_id, event.cost)

        if isinstance(event, paddle.event.EndPass):
            if event.pass_id % 10 == 0:
                filename = './output/model/params_pass_%d_%s.tar' % (event.pass_id, trainer_id_str)
                with open(filename, "w") as f:
                    trainer.save_parameter_to_tar(f)
            test_reader = paddle.batch(
                    paddle.reader.shuffle(
                        cluster_data_reader(cluster_test_dir), buf_size = 500),
                        batch_size = 2)
            result = trainer.test(
                reader = test_reader,
                feeding = feeding)
            print "Test %d, Cost %f" % (event.pass_id, result.cost)

    # training
    trainer.train(
        reader = train_reader,
        feeding = feeding,
        event_handler = event_handler,
        num_passes = 30)


def infer(file_name):
    """
    model infer
    """
    trainer_id_str = os.getenv("PADDLE_TRAINER_ID", "0")
    # inference
    test_reader = paddle.batch(
            paddle.reader.shuffle(
                cluster_data_reader(cluster_test_dir), buf_size = 500),
                batch_size = 1)
    infer_data = []
    infer_data_label = []
    for ins in test_reader():
        infer_data.append(ins[0][:-1])
        infer_data_label.append(ins[0][-1:])
    # load parameters from tar file.
    # users can remove the comments and change the model name
    with open(file_name, 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)
    # load topology net from pkl file
    topology_fp = open('./output/inference_topology.pkl', "r")
    inferer = paddle.inference.Inference(parameters=parameters, fileobj=topology_fp)
    probs = inferer.infer(input=infer_data)
    wfp = open("./output/predictions/part-{0:0>5d}".format(int(trainer_id_str)), "w")
    for i in xrange(len(probs)):
        wfp.write("i=%d, label=%s, predict=%s\n" % (i,
                str(infer_data_label[i][0][0]), str(probs[i][0])))
        print i, infer_data_label[i][0][0], probs[i][0]
    wfp.close()
    topology_fp.close()

if __name__ == '__main__':
    model_dir="./output/model"
    prediction_dir="./output/predictions"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
    train()
    infer("./output/model/params_pass_20_0.tar")
