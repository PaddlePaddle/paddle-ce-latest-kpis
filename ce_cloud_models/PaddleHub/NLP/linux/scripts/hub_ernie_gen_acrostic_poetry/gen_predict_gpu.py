#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  gen_predict_cpu.py
  * @date  2021/5/27 10:57 AM
  * @brief 
  *
  **************************************************************************/
"""
import paddlehub as hub

module = hub.Module(name="ernie_gen_acrostic_poetry")
test_texts = ['我喜欢你']
expect = [[
    '我方治地种秋芳，喜见新花照眼黄。欢友相逢头白尽，你缘何事得先尝。', '我今解此如意珠，喜汝为我返魂无。欢声百里镇如席，你若来时我自有。',
    '我今解此如意珠，喜汝为我返魂无。欢声百里镇如席，你若来时我自孤。', '我今解此如意珠，喜汝为我返魂无。欢声百里镇如席，你若来时我自如。',
    '我方治地种秋芳，喜见新花照眼黄。欢友相逢头白尽，你缘何事苦生凉。'
]]
results = module.generate(texts=test_texts, use_gpu=True, beam_width=5)
assert expect == results
