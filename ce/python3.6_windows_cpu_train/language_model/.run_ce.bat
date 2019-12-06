@echo off
set CPU_NUM=12
python train.py --data_path data/simple-examples/data/  --model_type small --use_gpu False --rnn_model static --max_epoch 1 --enable_ce | python _ce.py
