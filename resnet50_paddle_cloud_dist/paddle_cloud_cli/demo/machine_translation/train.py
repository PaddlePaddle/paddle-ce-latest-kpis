#!/bin/env python
# -*- encoding:utf-8 -*-
"""
machine translation demo
refer to:
https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_machine_translation.py
"""

import os
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.fluid.layers as pd
from paddle.fluid.executor import Executor

import paddlecloud.visual_util as visualdl

dict_size = 30000
source_dict_dim = target_dict_dim = dict_size
hidden_dim = 32
word_dim = 16
batch_size = 2
max_length = 8
topk_size = 50
trg_dic_size = 10000
beam_size = 2

decoder_size = hidden_dim

cluster_train_dir = "./train_data"
cluster_test_dir = "./test_data"

model_dir = "./output/model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


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
                        ins = line.split('\t')
                        src_ids = [int(w) for w in ins[0].split(',')]
                        trg_ids = [int(w) for w in ins[1].split(',')]
                        trg_ids_next = [int(w) for w in ins[2].split(',')]
                        yield src_ids, trg_ids, trg_ids_next
    return data_reader


def encoder(is_sparse):
    """
    encoder
    """
    src_word_id = pd.data(
        name="src_word_id", shape=[1], dtype='int64', lod_level=1)
    src_embedding = pd.embedding(
        input=src_word_id,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr=fluid.ParamAttr(name='vemb'))

    fc1 = pd.fc(input=src_embedding, size=hidden_dim * 4, act='tanh')
    lstm_hidden0, lstm_0 = pd.dynamic_lstm(input=fc1, size=hidden_dim * 4)
    encoder_out = pd.sequence_last_step(input=lstm_hidden0)
    return encoder_out


def decoder_train(context, is_sparse):
    """
    # decoder
    """
    trg_language_word = pd.data(
        name="target_language_word", shape=[1], dtype='int64', lod_level=1)
    trg_embedding = pd.embedding(
        input=trg_language_word,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr=fluid.ParamAttr(name='vemb'))

    rnn = pd.DynamicRNN()
    with rnn.block():
        current_word = rnn.step_input(trg_embedding)
        pre_state = rnn.memory(init=context)
        current_state = pd.fc(input=[current_word, pre_state],
                              size=decoder_size,
                              act='tanh')

        current_score = pd.fc(input=current_state,
                              size=target_dict_dim,
                              act='softmax')
        rnn.update_memory(pre_state, current_state)
        rnn.output(current_score)

    return rnn()


def decoder_decode(context, is_sparse):
    """
    decoder_decode
    """
    init_state = context
    array_len = pd.fill_constant(shape=[1], dtype='int64', value=max_length)
    counter = pd.zeros(shape=[1], dtype='int64', force_cpu=True)

    # fill the first element with init_state
    state_array = pd.create_array('float32')
    pd.array_write(init_state, array=state_array, i=counter)

    # ids, scores as memory
    ids_array = pd.create_array('int64')
    scores_array = pd.create_array('float32')

    init_ids = pd.data(name="init_ids", shape=[1], dtype="int64", lod_level=2)
    init_scores = pd.data(
        name="init_scores", shape=[1], dtype="float32", lod_level=2)

    pd.array_write(init_ids, array=ids_array, i=counter)
    pd.array_write(init_scores, array=scores_array, i=counter)

    cond = pd.less_than(x=counter, y=array_len)

    while_op = pd.While(cond=cond)
    with while_op.block():
        pre_ids = pd.array_read(array=ids_array, i=counter)
        pre_state = pd.array_read(array=state_array, i=counter)
        pre_score = pd.array_read(array=scores_array, i=counter)

        # expand the lod of pre_state to be the same with pre_score
        pre_state_expanded = pd.sequence_expand(pre_state, pre_score)

        pre_ids_emb = pd.embedding(
            input=pre_ids,
            size=[dict_size, word_dim],
            dtype='float32',
            is_sparse=is_sparse)

        # use rnn unit to update rnn
        current_state = pd.fc(input=[pre_state_expanded, pre_ids_emb],
                              size=decoder_size,
                              act='tanh')
        current_state_with_lod = pd.lod_reset(x=current_state, y=pre_score)
        # use score to do beam search
        current_score = pd.fc(input=current_state_with_lod,
                              size=target_dict_dim,
                              act='softmax')
        topk_scores, topk_indices = pd.topk(current_score, k=50)
        selected_ids, selected_scores = pd.beam_search(
            pre_ids, topk_indices, topk_scores, beam_size, end_id=10, level=0)

        pd.increment(x=counter, value=1, in_place=True)

        # update the memories
        pd.array_write(current_state, array=state_array, i=counter)
        pd.array_write(selected_ids, array=ids_array, i=counter)
        pd.array_write(selected_scores, array=scores_array, i=counter)

        pd.less_than(x=counter, y=array_len, cond=cond)

    translation_ids, translation_scores = pd.beam_search_decode(
        ids=ids_array, scores=scores_array)

    # return init_ids, init_scores

    return translation_ids, translation_scores


def set_init_lod(data, lod, place):
    """
    set_init_lod
    """
    res = fluid.LoDTensor()
    res.set(data, place)
    res.set_lod(lod)
    return res


def to_lodtensor(data, place):
    """
    to_lodtensor
    """
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def train_main(use_cuda, is_sparse, is_local=True):
    """
    train_main
    """
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    training_role = os.getenv("TRAINING_ROLE", "TRAINER")
    if training_role == "PSERVER":
        place = fluid.CPUPlace()

    context = encoder(is_sparse)
    rnn_out = decoder_train(context, is_sparse)
    label = pd.data(
        name="target_language_next_word", shape=[1], dtype='int64', lod_level=1)
    cost = pd.cross_entropy(input=rnn_out, label=label)
    avg_cost = pd.mean(cost)

    optimizer = fluid.optimizer.Adagrad(
        learning_rate=1e-4,
        regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.1))
    optimize_ops, params_grads = optimizer.minimize(avg_cost)

    exe = Executor(place)

    def train_loop(main_program):
        """
        train_loop
        """
        train_data = paddle.batch(
            paddle.reader.shuffle(
                cluster_data_reader(cluster_train_dir), buf_size=1000),
                batch_size=batch_size)
        exe.run(framework.default_startup_program())

        batch_id = 0
        for pass_id in xrange(1):
            for data in train_data():
                word_data = to_lodtensor(map(lambda x: x[0], data), place)
                trg_word = to_lodtensor(map(lambda x: x[1], data), place)
                trg_word_next = to_lodtensor(map(lambda x: x[2], data), place)
                outs = exe.run(main_program,
                               feed={
                                   'src_word_id': word_data,
                                   'target_language_word': trg_word,
                                   'target_language_next_word': trg_word_next
                               },
                               fetch_list=[avg_cost])
                avg_cost_val = np.array(outs[0])
                print('pass_id=' + str(pass_id) + ' batch=' + str(batch_id) +
                      " avg_cost=" + str(avg_cost_val))
                visualdl.show_fluid_trend('epoch', \
                        pass_id, batch_id, '{\"avg_loss\":%d}' % (avg_cost_val[0]))
                if batch_id > 10:
                    break
                batch_id += 1
            visualdl.show_fluid_trend('batch', \
                    pass_id, '{\"avg_loss\":%d}' % (avg_cost_val[0]))

    if is_local:
        train_loop(framework.default_main_program())
    else:
        port = os.getenv("PADDLE_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_PSERVERS")  # ip,ip...
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
        trainers = int(os.getenv("PADDLE_TRAINERS_NUM", "0"))
        current_endpoint = os.getenv("POD_IP") + ":" + port
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        t = fluid.DistributeTranspiler()
        t.transpile(
            optimize_ops,
            params_grads,
            trainer_id,
            pservers=pserver_endpoints,
            trainers=trainers)
        if training_role == "PSERVER":
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            train_loop(t.get_trainer_program())


if __name__ == '__main__':
    use_cuda = os.getenv("PADDLE_USE_GPU", "0") == "1"
    is_local = os.getenv("PADDLE_IS_LOCAL", "0") == "1"
    train_main(use_cuda=use_cuda, is_sparse=True, is_local=is_local)

