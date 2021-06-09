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

module = hub.Module(name="transformer_tts_ljspeech")

# Predict sentiment label
test_texts = [
    "Life was like a box of chocolates, you never know what you're gonna get."
]
wavs, sample_rate = module.synthesize(
    texts=test_texts, use_gpu=True, vocoder="waveflow")
for index, wav in enumerate(wavs):
    sf.write(f"{index}.wav", wav, sample_rate)
    if not os.path.exists(f"{index}.wav"):
        raise Exception(f"{index}.wav" + 'does not exist!!!!!!')
