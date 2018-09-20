import os
import time
import math
import numpy as np

import paddle
import paddle.fluid as fluid
import argparse
import reader
from network_conf import ner_net
from utils import logger, load_dict
from utils_extend import to_lodtensor, get_embedding
from continuous_evaluation import *

def parse_args():
    parser = argparse.ArgumentParser("sequence_tagging_for_ner model benchmark.")
    parser.add_argument(
        '--gpu_card_num', type=int, default=1, help='gpu card num used.')

    args = parser.parse_args()
    return args

def test(exe, chunk_evaluator, inference_program, test_data, place):
    chunk_evaluator.reset(exe)
    for data in test_data():
        word = to_lodtensor(map(lambda x: x[0], data), place)
        mark = to_lodtensor(map(lambda x: x[1], data), place)
        target = to_lodtensor(map(lambda x: x[2], data), place)
        acc = exe.run(inference_program,
                      feed={"word": word,
                            "mark": mark,
                            "target": target})
    return chunk_evaluator.eval(exe)


def main(train_data_file, test_data_file, vocab_file, target_file, emb_file,
         model_save_dir, num_passes, use_gpu, parallel):

    args = parse_args()
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    BATCH_SIZE = 200
    word_dict = load_dict(vocab_file)
    label_dict = load_dict(target_file)

    word_vector_values = get_embedding(emb_file)

    word_dict_len = len(word_dict)
    label_dict_len = len(label_dict)

    avg_cost, feature_out, word, mark, target = ner_net(
        word_dict_len, label_dict_len, parallel)

    inference_program = fluid.default_main_program().clone(for_test=True)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=1e-3)
    sgd_optimizer.minimize(avg_cost)

    crf_decode = fluid.layers.crf_decoding(
        input=feature_out, param_attr=fluid.ParamAttr(name='crfw'))

    chunk_evaluator = fluid.evaluator.ChunkEvaluator(
        input=crf_decode,
        label=target,
        chunk_scheme="IOB",
        num_chunk_types=int(math.ceil((label_dict_len - 1) / 2.0)))

    train_reader = paddle.batch(
            reader.data_reader(train_data_file, word_dict, label_dict),
        batch_size=BATCH_SIZE, drop_last=False)
    test_reader = paddle.batch(
            reader.data_reader(test_data_file, word_dict, label_dict),
        batch_size=BATCH_SIZE, drop_last=False)

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    feeder = fluid.DataFeeder(feed_list=[word, mark, target], place=place)
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())

    embedding_name = 'emb'
    embedding_param = fluid.global_scope().find_var(embedding_name).get_tensor(
    )
    embedding_param.set(word_vector_values, place)

    batch_id = 0
    total_time = 0.0
    for pass_id in xrange(num_passes):
        chunk_evaluator.reset(exe)
        start_time = time.time()
        for data in train_reader():
            cost, batch_precision, batch_recall, batch_f1_score = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[avg_cost] + chunk_evaluator.metrics)
            batch_id = batch_id + 1
        t1 = time.time()
        total_time += t1 - start_time
        pass_precision, pass_recall, pass_f1_score = chunk_evaluator.eval(exe)
        if pass_id == num_passes - 1:
            if args.gpu_card_num == 1:
                train_acc_kpi.add_record(pass_precision)
                pass_duration_kpi.add_record(total_time / num_passes)
            else:
                train_acc_kpi_card4.add_record(pass_precision)
                pass_duration_kpi_card4.add_record(total_time / num_passes)

        if pass_id % 100 == 0:
            print("[TrainSet] pass_id:" + str(pass_id) + " pass_precision:" +
                  str(pass_precision) + " pass_recall:" + str(
                      pass_recall) + " pass_f1_score:" + str(pass_f1_score))
        pass_precision, pass_recall, pass_f1_score = test(
            exe, chunk_evaluator, inference_program, test_reader, place)
        if pass_id % 100 == 0:
            print("[TestSet] pass_id:" + str(pass_id) + " pass_precision:" +
                  str(pass_precision) + " pass_recall:" + str(
                      pass_recall) + " pass_f1_score:" + str(pass_f1_score))

        #save_dirname = os.path.join(model_save_dir, "params_pass_%d" % pass_id)
        #fluid.io.save_inference_model(
        #    save_dirname, ['word', 'mark', 'target'], [crf_decode], exe)

    if args.gpu_card_num == 1:
        train_acc_kpi.persist()
        pass_duration_kpi.persist()
    else:
        train_acc_kpi_card4.persist()
        pass_duration_kpi_card4.persist()


if __name__ == "__main__":
    base_dir = "/root/.cache/paddle/dataset/data"
    main(
        train_data_file="data/train",
        test_data_file="data/test",
        vocab_file="%s/vocab.txt" % base_dir,
        target_file="data/target.txt",
        emb_file="%s/wordVectors.txt" % base_dir,
        model_save_dir="models",
        num_passes=2300,
        use_gpu=True,
        parallel=True)
