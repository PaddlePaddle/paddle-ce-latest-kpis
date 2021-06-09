#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  audio_finetune_gpu.py
  * @date  2021/5/20 8:38 PM
  * @brief 
  *
  **************************************************************************/
"""
import argparse
import ast

import paddle

import paddlehub as hub
from paddlehub.datasets import ESC50
import numpy as np
from paddlehub.env import MODULE_HOME
import os

import librosa

pwd = os.getcwd()
models_save = os.path.join(pwd, 'models_save')

parser = argparse.ArgumentParser(__doc__)
# parser.add_argument("--num_epoch", type=int, default=50, help="Number of epoches for fine-tuning.")
parser.add_argument(
    "--num_epoch",
    type=int,
    default=10,
    help="Number of epoches for fine-tuning.")
parser.add_argument(
    "--use_gpu",
    type=ast.literal_eval,
    default=True,
    help="Whether use GPU for fine-tuning, input should be True or False")
parser.add_argument(
    "--learning_rate",
    type=float,
    default=5e-5,
    help="Learning rate used to train with warmup.")
parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="Total examples' number in batch for training.")
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default=os.path.join(models_save),
    help="Directory to model checkpoint")
# parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint every n epoch.")
parser.add_argument(
    "--save_interval",
    type=int,
    default=2,
    help="Save checkpoint every n epoch.")
args = parser.parse_args()

if __name__ == "__main__":
    model = hub.Module(
        name='panns_cnn10', task='sound-cls', num_class=ESC50.num_class)

    train_dataset = ESC50(mode='train')
    dev_dataset = ESC50(mode='dev')

    optimizer = paddle.optimizer.AdamW(
        learning_rate=args.learning_rate, parameters=model.parameters())

    trainer = hub.Trainer(
        model,
        optimizer,
        checkpoint_dir=args.checkpoint_dir,
        use_gpu=args.use_gpu)
    trainer.train(
        train_dataset,
        epochs=args.num_epoch,
        batch_size=args.batch_size,
        eval_dataset=dev_dataset,
        save_interval=args.save_interval, )
