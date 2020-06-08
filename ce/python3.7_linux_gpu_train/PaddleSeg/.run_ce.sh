#!/bin/bash

model_list='deeplabv3p icnet unet pspnet hrnet fastscnn'
for model in ${model_list}
do

export CUDA_VISIBLE_DEVICES=0
python pdseg/train.py --use_gpu --enable_ce --cfg ./${model}.yaml BATCH_SIZE 2 1>log_${model}_card1 
cat log_${model}_card1 | grep "epoch" | tail -1 | tr '=' ' ' | awk '{print "kpis\t""'$model'""_loss_card1\t"$8"\nkpis\t""'$model'""_speed_card1\t"$10}' | python _ce.py

export CUDA_VISIBLE_DEVICES=0,1,2,3
python pdseg/train.py --use_gpu --enable_ce --cfg ./${model}.yaml 1>log_${model}_card4
cat log_${model}_card4 | grep "epoch" | tail -1 | tr '=' ' ' | awk '{print "kpis\t""'$model'""_loss_card4\t"$8"\nkpis\t""'$model'""_speed_card4\t"$10}' | python _ce.py

#eval
python pdseg/eval.py --use_gpu --cfg ./${model}.yaml >${model}_eval
if [ $? -ne 0 ];then
    echo -e "${model},eval,FAIL"
else
    echo -e "${model},eval,SUCCESS"
fi

#vis
python pdseg/vis.py --use_gpu --cfg ./${model}.yaml >${model}_infer
if [ $? -ne 0 ];then
    echo -e "${model},infer,FAIL"
else
    echo -e "${model},infer,SUCCESS"
fi

#export
python pdseg/export_model.py --cfg ./${model}.yaml >${model}_export
if [ $? -ne 0 ];then
    echo -e "${model},export,FAIL"
else
    echo -e "${model},export,SUCCESS"
fi
done

