from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import nose.tools as tools
import paddle
import paddle.fluid as fluid
import reader_cv2 as reader
sys.path.append("/models/PaddleCV/image_classification")
import models
import utils
from utils.utility import print_arguments, check_gpu


class Arguments(object):
    use_gpu = False
    class_dim = 1000
    image_shape = "3,224,224"
    pretrained_model = None
    model = None
    save_inference = True
    resize_short_size = 256
    inference_model_root = "/models/pretrained_model_dir/inference_model/"
    data_dir = "./data/ILSVRC2012"
    test_batch_size = 1
    test_sample_nums = 100


def get_norm_result(args):
    # parameters from arguments
    class_dim = args.class_dim
    model_name = args.model
    save_inference = args.save_inference
    pretrained_model = args.pretrained_model
    image_shape = [int(m) for m in args.image_shape.split(",")]
    model_list = [m for m in dir(models) if "__" not in m]
    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')

    # model definition
    model = models.__dict__[model_name]()
    if model_name == "GoogleNet":
        out, _, _ = model.net(input=image, class_dim=class_dim)
    else:
        out = model.net(input=image, class_dim=class_dim)
        out = fluid.layers.softmax(out)

    test_program = fluid.default_main_program().clone(for_test=True)

    fetch_list = [out.name]

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load pretrain model
    if len(pretrained_model) != 1:
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))
        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    feeder = fluid.DataFeeder(place=place, feed_list=[image])

    if save_inference:
        feed_var_names = ['image']
        target_vars = out
        fluid.io.save_inference_model(os.path.join(args.inference_model_root, args.model),
                                      feeded_var_names=feed_var_names,
                                      target_vars=target_vars,
                                      main_program=test_program,
                                      executor=exe,
                                      model_filename='model',
                                      params_filename='params')

    test_batch_size = args.test_batch_size
    test_reader = reader.test(settings=args, batch_size=test_batch_size, data_dir=args.data_dir)
    TOPK = 1
    res = []
    for batch_id, data in enumerate(test_reader()):
        if batch_id == args.test_sample_nums:
            break

        result = exe.run(test_program,
                         fetch_list=fetch_list,
                         feed=feeder.feed(data))
        result = result[0][0]
        pred_label = np.argsort(result)[::-1][:TOPK]

        print("Test-{0}-score: {1}, class {2}"
              .format(batch_id, result[pred_label], pred_label))
        sys.stdout.flush()
        res += result.tolist()

    return res


def get_inference_result(args):
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()

    exe = fluid.Executor(place)
    [inference_program, feed_var_names, target_vars] = \
        fluid.io.load_inference_model(dirname=os.path.join(args.inference_model_root, args.model),
                                      executor=exe,
                                      model_filename='model',
                                      params_filename='params')
    test_program = inference_program.clone(for_test=True)

    test_batch_size = args.test_batch_size
    test_reader = reader.test(settings=args, batch_size=test_batch_size, data_dir=args.data_dir)
    feeder = fluid.DataFeeder(place=place, feed_list=[feed_var_names[0]])
    TOPK = 1
    res = []
    for batch_id, data in enumerate(test_reader()):
        if batch_id == args.test_sample_nums:
            break

        result = exe.run(test_program,
                         feed=feeder.feed(data),
                         fetch_list=target_vars)
        result = result[0][0]
        pred_label = np.argsort(result)[::-1][:TOPK]

        print("Test-{0}-score: {1}, class {2}"
              .format(batch_id, result[pred_label], pred_label))
        sys.stdout.flush()
        res += result.tolist()

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

    def test_resnet50(self):
        args = Arguments()
        args.pretrained_model = "/models/pretrain_models/ResNet50_pretrained"
        args.model = "ResNet50"
        print_arguments(args)
        expect = get_norm_result(args)
        result = get_inference_result(args)
        assert len(expect) == len(result)
        for i in range(0, len(expect)):
            tools.assert_almost_equal(expect[i], result[i], delta=1e-5)

    def test_googlenet(self):
        args = Arguments()
        args.pretrained_model = "/models/pretrain_models/GoogleNet_pretrained"
        args.model = "GoogleNet"
        print_arguments(args)
        expect = get_norm_result(args)
        result = get_inference_result(args)
        assert len(expect) == len(result)
        for i in range(0, len(expect)):
            tools.assert_almost_equal(expect[i], result[i], delta=1e-5)
