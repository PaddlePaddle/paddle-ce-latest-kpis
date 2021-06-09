#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  textmatching_finetune_gpu.py
  * @date  2021/4/21 2:07 PM
  * @brief 
  *
  **************************************************************************/
"""
import paddle
import os
import shutil
import paddlehub as hub
from paddlehub.datasets import LCQMC

pwd = os.getcwd()
models_save = os.path.join(pwd, 'models_save')
if os.path.exists(models_save):
    shutil.rmtree(models_save)
os.mkdir(os.path.join(pwd, 'models_save'))

model = hub.Module(
    name='bert-base-multilingual-cased', version='2.0.2', task='text-matching')

train_dataset = LCQMC(
    tokenizer=model.get_tokenizer(), max_seq_len=128, mode='train')
dev_dataset = LCQMC(
    tokenizer=model.get_tokenizer(), max_seq_len=128, mode='dev')
test_dataset = LCQMC(
    tokenizer=model.get_tokenizer(), max_seq_len=128, mode='test')

# optimizer = paddle.optimizer.AdamW(learning_rate=5e-5, parameters=model.parameters())
optimizer = paddle.optimizer.Adam(
    learning_rate=1e-5, parameters=model.parameters())
trainer = hub.Trainer(
    model,
    optimizer,
    checkpoint_dir=os.path.join(models_save, 'bert-base-multilingual-cased'),
    use_gpu=True)

trainer.train(
    dev_dataset,
    epochs=2,
    batch_size=16,
    eval_dataset=dev_dataset,
    save_interval=1, )
trainer.evaluate(test_dataset, batch_size=16)
