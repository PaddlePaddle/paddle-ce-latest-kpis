#!/bin/bash
# gpu1
CUDA_VISIBLE_DEVICES=3 python mmoe_train.py --use_gpu True >mmoe_gpu1_T 2>&1
cat mmoe_gpu1_T |awk -F ':| ' 'END{print "kpis\ttrain_loss_gpu1\t"$4}' |tr -d '[|]|,' | python _ce.py
# gpu8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python mmoe_train.py --use_gpu True >mmoe_gpu8_T 2>&1
cat mmoe_gpu8_T |awk -F ':| ' 'END{print "kpis\ttrain_loss_gpu8\t"$4}' |tr -d '[|]|,' | python _ce.py
