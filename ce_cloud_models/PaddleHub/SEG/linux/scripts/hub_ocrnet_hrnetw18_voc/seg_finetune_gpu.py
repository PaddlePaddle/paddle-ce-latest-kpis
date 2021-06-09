#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  seg_finetune_gpu.py
  * @date  2021/5/8 2:25 PM
  * @brief 
  *
  **************************************************************************/
"""
import paddle
import numpy as np
import paddlehub as hub
from paddlehub.finetune.trainer import Trainer
from paddlehub.datasets import OpticDiscSeg
from paddlehub.vision.segmentation_transforms import Compose, Resize, Normalize
from paddlehub.vision.utils import ConfusionMatrix
import cv2
import os
import shutil

pwd = os.getcwd()
models_save = os.path.join(pwd, 'models_save')
if os.path.exists(models_save):
    shutil.rmtree(models_save)
os.mkdir(os.path.join(pwd, 'models_save'))

train_transforms = Compose([Resize(target_size=(512, 512)), Normalize()])
eval_transforms = Compose([Normalize()])
train_reader = OpticDiscSeg(train_transforms)
eval_reader = OpticDiscSeg(eval_transforms, mode='val')

model = hub.Module(name='ocrnet_hrnetw18_voc', num_classes=2)
scheduler = paddle.optimizer.lr.PolynomialDecay(
    learning_rate=0.01, decay_steps=1000, power=0.9, end_lr=0.0001)
optimizer = paddle.optimizer.Momentum(
    learning_rate=scheduler, parameters=model.parameters())
trainer = Trainer(
    model,
    optimizer,
    checkpoint_dir=os.path.join(models_save, 'ocrnet_hrnetw18_voc'),
    use_gpu=True)
trainer.train(
    train_reader, epochs=2, batch_size=8, log_interval=1, save_interval=2)

cfm = ConfusionMatrix(eval_reader.num_classes, streaming=True)
model.eval()
for imgs, labels in eval_reader:
    imgs = imgs[np.newaxis, :, :, :]
    preds = model(paddle.to_tensor(imgs))[0]
    preds = paddle.argmax(preds, axis=1, keepdim=True).numpy()
    labels = labels[np.newaxis, :, :, :]
    ignores = labels != eval_reader.ignore_index
    cfm.calculate(preds, labels, ignores)
_, miou = cfm.mean_iou()
print('miou: {:.4f}'.format(miou))
