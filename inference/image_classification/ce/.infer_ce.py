#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
import paddle.fluid as fluid
import numpy as np
import os
import sys

sys.path.append("..")
import argparse
import functools
from utils.utility import add_arguments, print_arguments
import models

parser = argparse.ArgumentParser(description=__doc__)

add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('pretrained_model_dir', str, "", "The pretrained model dir, if no setted, random the model weight")
add_arg('model', str, "ResNet50", "Set the network to use.")
add_arg('inference_model_root', str, "/models/pretrained_model_dir/inference_model/", "the saved inference model dir")
add_arg('data_path', str, "./inference_model/", "the saved inference model dir")
add_arg('use_gpu', bool, True, "Whether to use GPU or not.")
add_arg('class_dim', int, 1000, "Class number.")
add_arg('image_shape', str, "3,224,224", "Input image size")
add_arg('batch_size', int, 1, "Input data batch_size")


def get_norm_result(args, input_data):
    """
    use load_vars to load pretrained model and infer
    use save_inference_model to save models to another file

    Args:
    args: input argument
    input_data: a np.array with dimension

    Returns:
    result[0][0]: a list contains infer results
    """
    model_name = args.model

    model = models.__dict__[model_name]()

    image_shape = [3, 224, 224]
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')

    net = model.net(input=image, class_dim=1000)
    net = fluid.layers.softmax(net)

    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)
    test_program = fluid.default_main_program().clone(for_test=True)
    exe.run(fluid.default_startup_program())

    # load pretrain model
    if len(args.pretrained_model_dir) != 1:
        def if_exist(var):
            return os.path.exists(os.path.join(args.pretrained_model_dir, var.name))
        fluid.io.load_vars(exe, args.pretrained_model_dir, predicate=if_exist)

    fetch_list = [net]
    result = exe.run(test_program,
                     fetch_list=fetch_list,
                     feed={'image': input_data})

    feed_var_names = ['image']
    target_vars = net
    fluid.io.save_inference_model(os.path.join(args.inference_model_root, args.model),
                                  feeded_var_names=feed_var_names,
                                  target_vars=target_vars,
                                  main_program=test_program,
                                  executor=exe,
                                  model_filename='model',
                                  params_filename='params')
    return result[0][0]


def get_inference_result(args, input_data):
    """
    use load_inference_model to load model and infer

    Args:
    args: input argument
    input_data: a np.array with dimension

    Returns:
    result[0][0]: a list contains infer results
    """
    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)
    [inference_program, feed_var_names, target_vars] = \
        fluid.io.load_inference_model(dirname=os.path.join(args.inference_model_root, args.model),
                                      executor=exe,
                                      model_filename='model',
                                      params_filename='params')
    test_program = inference_program.clone(for_test=True)
    result = exe.run(test_program,
                     feed={feed_var_names[0]: input_data},
                     fetch_list=target_vars)

    return result[0][0]


def load_fake_data(batch_size=1, channels=3, height=224, width=224):
    """
    load test fake data
    """
    data = np.ones((batch_size, channels, height, width)).astype('float32')
    return data


def main():
    """
    main function
    """
    args = parser.parse_args()
    image_shape = [int(m) for m in args.image_shape.split(",")]
    input_data = load_fake_data(args.batch_size,
                                image_shape[0],
                                image_shape[1],
                                image_shape[2])
    print_arguments(args)
    get_norm_result(args, input_data)
    get_inference_result(args, input_data)


if __name__ == '__main__':
    main()

