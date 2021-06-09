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

module = hub.Module(name="ernie_gen_couplet")
test_texts = ["人增福寿年增岁", "风吹云乱天垂泪"]
expect = [['春满乾坤喜满门', '竹报平安梅报春', '春满乾坤福满门', '春满乾坤酒满樽', '春满乾坤喜满家'],
          ['雨打花残地痛心', '雨打花残地皱眉', '雨打花残地动容', '雨打霜欺地动容', '雨打花残地洒愁']]
results = module.generate(texts=test_texts, use_gpu=False, beam_width=5)
assert expect == results
