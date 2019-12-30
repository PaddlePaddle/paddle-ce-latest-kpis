export CUDA_VISIBLE_DEVICES=0

python  train.py \
    --data_path data/simple-examples/data/ \
    --model_type small \
    --use_gpu True \
    --rnn_model static \
    --enable_ce 1> log_static
cat log_static | python _ce.py

python  train.py \
    --data_path data/simple-examples/data/ \
    --model_type small \
    --use_gpu True \
    --rnn_model padding \
    --enable_ce 1> log_padding
cat log_padding | python _ce.py
