@echo off
set CUDA_VISIBLE_DEVICES=0
python train.py --data_path data/simple-examples/data/  --model_type small --use_gpu True --rnn_model static --max_epoch 1 --enable_ce | python _ce.py
python train.py --data_path data/simple-examples/data/  --model_type small --use_gpu True --rnn_model padding --max_epoch 1 --enable_ce | python _ce.py
