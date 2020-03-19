#!/bin/bash
# This file is only used for continuous evaluation.

export CPU_NUM=1
python train.py --model_net CycleGAN --dataset cityscapes  --net_G resnet_9block --g_base_dim 32 --net_D basic --norm_type batch_norm --image_size 286 --crop_size 256 --output ./output/cyclegan/  --epoch 1 --enable_ce --shuffle False --run_test True --save_checkpoints True --use_gpu False  | python _ce.py
# infer
python infer.py --init_model output/cyclegan/checkpoints/0/ --dataset_dir data/cityscapes/ --image_size 256 --n_samples 20 --crop_size 256 --input_style B --test_list ./data/cityscapes/testB.txt --model_net CycleGAN --net_G resnet_9block --g_base_dims 32 --output ./infer_result/cyclegan/ --use_gpu false > $log_path/cyclegan_I.log 2>&1
if [ $? -ne 0 ];then↩
         mv ${log_path}/cyclegan_I.log ${log_path}/FAIL/cyclegan_I.log↩
         echo -e "cyclegan,infer,FAIL" >>${log_path}/result.log;↩
else↩
         mv ${log_path}/cyclegan_I.log ${log_path}/SUCCESS/cyclegan_I.log↩
         echo -e "cyclegan,infer,SUCCESS" >>${log_path}/result.log↩
fi↩
