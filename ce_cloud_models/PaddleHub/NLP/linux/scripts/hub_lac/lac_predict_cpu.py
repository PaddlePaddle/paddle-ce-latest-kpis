#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  lac_predict_cpu.py
  * @date  2021/5/26 10:48 AM
  * @brief 
  *
  **************************************************************************/
"""
import paddlehub as hub

lac = hub.Module(name="lac")
test_text = ["今天是个好日子", "天气预报说今天要下雨", "下一班地铁马上就要到了"]
expect0 = [{
    'word': ['今天', '是', '个', '好日子'],
    'tag': ['TIME', 'v', 'q', 'n']
}, {
    'word': ['天气预报', '说', '今天', '要', '下雨'],
    'tag': ['n', 'v', 'TIME', 'v', 'v']
}, {
    'word': ['下', '一班', '地铁', '马上', '就要', '到', '了'],
    'tag': ['f', 'm', 'n', 'd', 'v', 'v', 'xc']
}]
# expect1 = {'n': '普通名词', 'f': '方位名词', 's': '处所名词', 't': '时间',
#            'nr': '人名', 'ns': '地名', 'nt': '机构名', 'nw': '作品名',
#            'nz': '其他专名', 'v': '普通动词', 'vd': '动副词', 'vn': '名动词',
#            'd': '副词', 'm': '数量词', 'q': '量词', 'r': '代词',
#            'p': '介词', 'c': '连词', 'u': '助词', 'xc': '其他虚词',
#            'w': '标点符号', 'PER': '人名', 'LOC': '地名', 'ORG': '机构名', 'TIME': '时间'}
results0 = lac.cut(text=test_text,
                   use_gpu=False,
                   batch_size=1,
                   return_tag=True)
# results1 = lac.get_tags()
assert results0 == expect0
# assert results1 == expect1
# print(lac.get_tags())
