#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  senta_predict_cpu.py
  * @date  2021/5/26 11:11 AM
  * @brief 
  *
  **************************************************************************/
"""
import paddlehub as hub

senta = hub.Module(name="senta_lstm")
test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]
expect0 = [{
    'text': ['这家餐厅很好吃', '这部电影真的很差劲'],
    'sentiment_label': 0,
    'sentiment_key': 'negative',
    'positive_probs': 0.376,
    'negative_probs': 0.624
}, {
    'text': ['这家餐厅很好吃', '这部电影真的很差劲'],
    'sentiment_label': 0,
    'sentiment_key': 'negative',
    'positive_probs': 0.376,
    'negative_probs': 0.624
}]
results0 = senta.sentiment_classify(
    texts=[test_text, test_text], use_gpu=False, batch_size=2)
# print(results0)
assert expect0 == results0
expect1 = {'positive': 1, 'negative': 0}
results1 = senta.get_labels()
assert expect1 == results1
results2 = senta.get_vocab_path()
