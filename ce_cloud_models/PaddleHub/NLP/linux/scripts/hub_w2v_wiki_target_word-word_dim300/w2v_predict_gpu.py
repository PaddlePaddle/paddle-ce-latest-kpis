#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  w2v_predict_gpu.py
  * @date  2021/6/8 8:18 PM
  * @brief 
  *
  **************************************************************************/
"""
import paddlehub as hub

embedding = hub.Module(name='w2v_wiki_target_word-word_dim300')
# 获取单词的embedding
embedding.search("中国")
# 计算两个词向量的余弦相似度
embedding.cosine_sim("中国", "美国")
# 计算两个词向量的内积
embedding.dot("中国", "美国")
