#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  sim_predict_cpu.py
  * @date  2021/5/27 11:36 AM
  * @brief 
  *
  **************************************************************************/
"""
import paddlehub as hub

simnet_bow = hub.Module(name="simnet_bow")
test_text_1 = ["这道题太难了", "这道题太难了", "这道题太难了"]
test_text_2 = ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]
test_text = [test_text_1, test_text_2]
expect = [{
    'text_1': '这道题太难了',
    'text_2': '这道题是上一年的考题',
    'similarity': 0.689
}, {
    'text_1': '这道题太难了',
    'text_2': '这道题不简单',
    'similarity': 0.855
}, {
    'text_1': '这道题太难了',
    'text_2': '这道题很有意思',
    'similarity': 0.8166
}]
results = simnet_bow.similarity(texts=test_text, use_gpu=False, batch_size=2)
assert expect == results
