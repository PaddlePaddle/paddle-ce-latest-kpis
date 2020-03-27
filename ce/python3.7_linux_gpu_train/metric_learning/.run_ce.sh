#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
python train_elem.py --model=ResNet50 --test_iter_step=10 --loss_name=arcmargin --arc_scale=80.0 --arc_margin=0.15 --arc_easy_margin=False --total_iter_num=10 --model_save_dir=output --save_iter_step=1 --enable_ce=True 1>log
cat log | python _ce.py

#eval
python eval.py \
       --model=ResNet50 \
       --batch_size=50 \
       --pretrained_model=output >eval
if [ $? -ne 0 ];then
       echo -e "metric,eval,FAIL"
else
       echo -e "metric,eval,SUCCESS"
fi

#infer
python infer.py \
       --model=ResNet50 \
       --batch_size=50 \
       --pretrained_model=output >infer
if [ $? -ne 0 ];then
       echo -e "metric,infer,FAIL"
else
       echo -e "metric,infer,SUCCESS"
fi
