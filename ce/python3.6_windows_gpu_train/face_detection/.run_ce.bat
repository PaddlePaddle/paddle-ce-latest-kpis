@echo off

set CUDA_VISIBLE_DEVICES=0
set ce_mode=1
python -u train.py --epoc_num=1 --use_multiprocess=False --batch_size=2 --pretrained_model=vgg_ilsvrc_16_fc_reduced --parallel=False --use_gpu True --enable_ce | python _ce.py

