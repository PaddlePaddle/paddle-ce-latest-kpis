#!/bin/bash

export CPU_NUM=10
python  train.py --train_path=data/train/sentence_file_*  --test_path=data/dev/sentence_file_* --vocab_path data/vocabulary_min5k.txt --learning_rate 0.2 --use_gpu False --all_train_tokens 35479 --max_epoch 1 --log_interval 5 --dev_interval 20 --local True --batch_size 32 --shuffle false --random_seed 100 --enable_ce| python _ce.py
