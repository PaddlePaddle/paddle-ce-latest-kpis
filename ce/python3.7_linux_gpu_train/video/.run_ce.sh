#!/bin/bash
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

model_list=(AttentionCluster AttentionLSTM NEXTVLAD STNET TSM TSN NONLOCAL BsnTem BMN ETS TALL)
config_list=(attention_cluster attention_lstm nextvlad stnet tsm tsn nonlocal bsn_tem bmn ets tall)
for((i=0;i<11;i++));
do
if [[ ${model_list[${i}]} == 'BsnTem' || ${model_list[${i}]} == 'TALL' ]];then
    export CUDA_VISIBLE_DEVICES=0
elif [[ ${model_list[${i}]} == 'BMN' || ${model_list[${i}]} == 'ETS' || ${model_list[${i}]} == 'NEXTVLAD' ]];then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
else
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi
python train.py \
       --model_name=${model_list[${i}]} \
       --config=./configs/${config_list[${i}]}.yaml \
       --epoch=1 \
       --log_interval=10 \
       --fix_random_seed=True >log_${config_list[${i}]} 2>&1

#set different loss index because of different log structure
if [[ ${model_list[${i}]} == 'STNET' || ${model_list[${i}]} == 'TSM' || ${model_list[${i}]} == 'TSN' || ${model_list[${i}]} == 'NONLOCAL' ]];then
   cat log_${config_list[${i}]} | grep "Epoch 0" | tail "-2" | head -1 | tr ',' ' ' | awk '{print "kpis\t""'${model_list[i]}'""_loss\t"$10}' | python _ce.py
else
   cat log_${config_list[${i}]} | grep "Epoch 0" | tail "-2" | head -1 | tr ',' ' ' | awk '{print "kpis\t""'${model_list[i]}'""_loss\t"$15}' | python _ce.py
fi
cat log_${config_list[${i}]} | grep "Epoch 0" | tail "-1" | tr ',' ' ' | awk '{print "kpis\t""'${model_list[i]}'""_time\t"$11}' | python _ce.py

done
