###!/bin/bash
####This file is only used for continuous evaluation.

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

if [ ! -d "/root/.cache/paddle/dataset/pascalvoc" ];then
    mkdir -p /root/.cache/paddle/dataset/pascalvoc
    ./data/pascalvoc/download.sh
    cp -r ./data/pascalvoc/. /home/.cache/paddle/dataset/pascalvoc
fi

cudaid=${object_detection_cudaid:=0}
export CUDA_VISIBLE_DEVICES=$cudaid
FLAGS_benchmark=true  python train.py --enable_ce=True --batch_size=64 --epoc_num=2 --data_dir=/root/.cache/paddle/dataset/pascalvoc/ 1> log_1card
cat log_1card | python _ce.py

cudaid=${object_detection_cudaid_m:=0,1,2,3}
export CUDA_VISIBLE_DEVICES=$cudaid
FLAGS_benchmark=true  python train.py --enable_ce=True --batch_size=64 --epoc_num=2 --data_dir=/root/.cache/paddle/dataset/pascalvoc/ 1> log_4cards
cat log_4cards | python _ce.py

#eval
python eval.py --dataset='pascalvoc' --model_dir='model/best_model' --data_dir='/root/.cache/paddle/dataset/pascalvoc/' --test_list='test.txt' --ap_version='11point' >eval
if [ $? -ne 0 ];then
    echo -e "ssd,eval,FAIL"
else
    echo -e "ssd,eval,SUCCESS"
fi

#infer
python infer.py --dataset='pascalvoc' --nms_threshold=0.45 --model_dir='model/best_model' --image_path='/root/.cache/paddle/dataset/pascalvoc/VOCdevkit/VOC2007/JPEGImages/009963.jpg' >infer
if [ $? -ne 0 ];then
    echo -e "ssd,infer,FAIL"
else
    echo -e "ssd,infer,SUCCESS"
fi
