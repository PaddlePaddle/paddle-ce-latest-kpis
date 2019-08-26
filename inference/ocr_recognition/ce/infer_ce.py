from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import numpy as np
import nose.tools as tools
import sys
sys.path.append("/models/PaddleCV/orc_recognition")
import data_reader
from utility import print_arguments
from utility import get_attention_feeder_for_infer, get_ctc_feeder_for_infer
from crnn_ctc_model import ctc_infer
from attention_model import attention_infer


class Arguments(object):
    model = "crnn_ctc"
    model_path = "/models/pretrain_models/ocr_ctc/ocr_ctc_params"
    input_images_dir = "/models/ctc_data/data/test_images/"
    input_images_list = "/models/ctc_data/data/test.list"
    dict = None
    use_gpu = False
    iterations = 5
    profile = False
    skip_batch_num = 0
    batch_size = 1
    inference_model_root = "/models/pretrained_model_dir/inference_model/"
    save_inference = True


def prune(words, sos, eos):
    """Remove unused tokens in prediction result."""
    start_index = 0
    end_index = len(words)
    if sos in words:
        start_index = np.where(words == sos)[0][0] + 1
    if eos in words:
        end_index = np.where(words == eos)[0][0]
    return words[start_index:end_index]


def get_norm_result(args):
    """OCR inference"""
    if args.model == "crnn_ctc":
        infer = ctc_infer
        get_feeder_data = get_ctc_feeder_for_infer
    else:
        infer = attention_infer
        get_feeder_data = get_attention_feeder_for_infer

    num_classes = data_reader.num_classes()
    data_shape = data_reader.data_shape()
    # define network
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    ids = infer(images, num_classes, use_cudnn=True if args.use_gpu else False)
    # data reader
    infer_reader = data_reader.inference(
        batch_size=args.batch_size,
        infer_images_dir=args.input_images_dir,
        infer_list_file=args.input_images_list,
        cycle=True if args.iterations > 0 else False,
        model=args.model)
    # prepare environment
    place = fluid.CPUPlace()
    if args.use_gpu:
        place = fluid.CUDAPlace(0)

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    test_program = fluid.default_main_program().clone(for_test=True)

    # load init model
    model_dir = args.model_path
    model_file_name = None
    if not os.path.isdir(args.model_path):
        model_dir = os.path.dirname(args.model_path)
        model_file_name = os.path.basename(args.model_path)
    fluid.io.load_params(exe, dirname=model_dir, filename=model_file_name)
    print("Init model from: %s." % args.model_path)

    # save inference model
    if args.save_inference:
        if args.model == 'crnn_ctc':
            feed_var_names = ['pixel']
        elif args.model == 'attention':
            feed_var_names = ['init_ids', 'pixel', 'init_scores']
        target_vars = ids
        fluid.io.save_inference_model(os.path.join(args.inference_model_root, args.model),
                                      feeded_var_names=feed_var_names,
                                      target_vars=target_vars,
                                      main_program=test_program,
                                      executor=exe,
                                      model_filename='model',
                                      params_filename='params')
    batch_times = []
    iters = 0
    res = []
    for data in infer_reader():
        feed_dict = get_feeder_data(data, place)
        if args.iterations > 0 and iters == args.iterations + args.skip_batch_num:
            break
        if iters < args.skip_batch_num:
            print("Warm-up itaration")
        if iters == args.skip_batch_num:
            profiler.reset_profiler()

        start = time.time()
        result = exe.run(test_program,
                         feed=feed_dict,
                         fetch_list=[ids],
                         return_numpy=False)
        indexes = prune(np.array(result[0]).flatten(), 0, 1)
        batch_time = time.time() - start
        fps = args.batch_size / batch_time
        batch_times.append(batch_time)

        print("Iteration %d, latency: %.5f s, fps: %f, result: %s" % (
            iters,
            batch_time,
            fps,
            indexes, ))

        iters += 1
        res += indexes.tolist()
    return res


def get_inference_result(args):
    """OCR get inference result"""
    if args.model == "crnn_ctc":
        get_feeder_data = get_ctc_feeder_for_infer
    else:
        get_feeder_data = get_attention_feeder_for_infer

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()

    exe = fluid.Executor(place)
    [inference_program, feed_var_names, target_vars] = \
        fluid.io.load_inference_model(dirname=os.path.join(args.inference_model_root, args.model),
                                      executor=exe,
                                      model_filename='model',
                                      params_filename='params')

    # data reader
    infer_reader = data_reader.inference(
        batch_size=args.batch_size,
        infer_images_dir=args.input_images_dir,
        infer_list_file=args.input_images_list,
        cycle=True if args.iterations > 0 else False,
        model=args.model)

    batch_times = []
    iters = 0
    res = []
    for data in infer_reader():
        feed_dict = get_feeder_data(data, place)
        if args.iterations > 0 and iters == args.iterations + args.skip_batch_num:
            break
        if iters < args.skip_batch_num:
            print("Warm-up itaration")
        if iters == args.skip_batch_num:
            profiler.reset_profiler()

        start = time.time()
        result = exe.run(inference_program,
                         feed=feed_dict,
                         fetch_list=target_vars,
                         return_numpy=False)
        indexes = prune(np.array(result[0]).flatten(), 0, 1)
        batch_time = time.time() - start
        fps = args.batch_size / batch_time
        batch_times.append(batch_time)

        print("Iteration %d, latency: %.5f s, fps: %f, result: %s" % (
            iters,
            batch_time,
            fps,
            indexes, ))

        iters += 1
        res += indexes.tolist()
    return res


class TestSaveLoadAPI(object):
    """
    Test Save Load inference model
    """

    def __init__(self):
        """
        __init__
        """
        pass

    def test_ctc(self):
        args = Arguments()
        print_arguments(args)
        expect = get_norm_result(args)
        result = get_inference_result(args)
        assert len(expect) == len(result)
        for i in range(0, len(expect)):
            tools.assert_almost_equal(expect[i], result[i], delta=1e-5)

    def test_attention(self):
        args = Arguments()
        args.model = "attention"
        args.model_path = "/models/pretrain_models/ocr_attention/ocr_attention_params"
        print_arguments(args)
        expect = get_norm_result(args)
        result = get_inference_result(args)
        assert len(expect) == len(result)
        for i in range(0, len(expect)):
            tools.assert_almost_equal(expect[i], result[i], delta=1e-5)
