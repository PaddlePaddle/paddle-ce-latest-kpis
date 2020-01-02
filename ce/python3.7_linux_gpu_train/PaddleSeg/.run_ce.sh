#!/bin/bash

model_list='deeplabv3p icnet unet pspnet hrnet'
for model in ${model_list}
do

export CUDA_VISIBLE_DEVICES=0
python pdseg/train.py --use_gpu --enable_ce --cfg ./${model}.yaml BATCH_SIZE 2 1>log_${model}_card1 
cat log_${model}_card1 | grep "epoch" | tail -1 | tr '=' ' ' | awk '{print "kpis\t""'$model'""_loss_card1\t"$8"\nkpis\t""'$model'""_speed_card1\t"$10}' | python _ce.py

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python pdseg/train.py --use_gpu --enable_ce --cfg ./${model}.yaml 1>log_${model}_card8 
cat log_${model}_card8 | grep "epoch" | tail -1 | tr '=' ' ' | awk '{print "kpis\t""'$model'""_loss_card8\t"$8"\nkpis\t""'$model'""_speed_card8\t"$10}' | python _ce.py

done



