# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  syn_predict_cpu.py
  * @date  2021/6/8 5:17 PM
  * @brief 
  *
  **************************************************************************/
"""
import paddlehub as hub
import soundfile as sf
import os

module = hub.Module(name="fastspeech_ljspeech")

# Predict sentiment label
test_texts = [
    'Simple as this proposition is, it is necessary to be stated',
    'Parakeet stands for Paddle PARAllel text-to-speech toolkit'
]
wavs, sample_rate = module.synthesize(texts=test_texts, use_gpu=True)
for index, wav in enumerate(wavs):
    sf.write(f"{index}.wav", wav, sample_rate)
    if not os.path.exists(f"{index}.wav"):
        raise Exception(f"{index}.wav" + 'does not exist!!!!!!')
