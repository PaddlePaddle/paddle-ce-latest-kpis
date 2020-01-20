#!/bin/bash
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

#model_list=(AttentionCluster AttentionLSTM NEXTVLAD STNET TSM TSN NONLOCAL CTCN BSN BMN ETS TALL)
#config_list=(attention_cluster attention_lstm nextvlad stnet tsm tsn nonlocal ctcn bsn_tem bmn ets tall)
model_list=(AttentionCluster AttentionLSTM NEXTVLAD STNET TSM TSN NONLOCAL CTCN)
config_list=(attention_cluster attention_lstm nextvlad stnet tsm tsn nonlocal ctcn)
for((i=0;i<8;i++));
do
BATCH_SIZE=16

#CTCN's batch_size is set as 2 to avoid out of memory
if [ ${model_list[${i}]} == 'CTCN' ];then
    BATCH_SIZE=2
fi

export CUDA_VISIBLE_DEVICES=0
if [ ${model_list[${i}]} == 'NEXTVLAD' ];then
    sed -i "s/num_gpus: 4/num_gpus: 1/g" configs/${config_list[${i}]}.yaml
else
    sed -i "s/num_gpus: 8/num_gpus: 1/g" configs/${config_list[${i}]}.yaml
fi

python train.py \
       --model_name=${model_list[${i}]} \
       --config=./configs/${config_list[${i}]}.yaml \
       --epoch=1 \
       --log_interval=10 \
       --batch_size=${BATCH_SIZE} \
       --fix_random_seed=True >log_${config_list[${i}]}_1card 2>&1
#set different loss index because of different log structure
if [[ ${model_list[${i}]} == 'AttentionCluster' || ${model_list[${i}]} == 'AttentionLSTM' || ${model_list[${i}]} == 'NEXTVLAD' || ${model_list[${i}]} == 'CTCN' ]];then
    cat log_${config_list[${i}]}_1card | grep "Epoch 0" | tail "-2" | head -1 | tr ',' ' ' | awk '{print "kpis\t""'${model_list[i]}'""_loss_card1\t"$11}' | python _ce.py
else
   cat log_${config_list[${i}]}_1card | grep "Epoch 0" | tail "-2" | head -1 | tr ',' ' ' | awk '{print "kpis\t""'${model_list[i]}'""_loss_card1\t"$10}' | python _ce.py     
fi
cat log_${config_list[${i}]}_1card | grep "Epoch 0" | tail "-1" | tr ',' ' ' | awk '{print "kpis\t""'${model_list[i]}'""_time_card1\t"$11}' | python _ce.py

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
sed -i "s/num_gpus: 1/num_gpus: 8/g" configs/${config_list[${i}]}.yaml
python train.py \
       --model_name=${model_list[${i}]} \
       --config=./configs/${config_list[${i}]}.yaml \
       --epoch=1 \
       --log_interval=10 \
       --fix_random_seed=True >log_${config_list[${i}]}_8cards 2>&1
if [[ ${model_list[${i}]} == 'AttentionCluster' || ${model_list[${i}]} == 'AttentionLSTM' || ${model_list[${i}]} == 'NEXTVLAD' || ${model_list[${i}]} == 'CTCN' ]];then
    cat log_${config_list[${i}]}_8cards | grep "Epoch 0" | tail "-2" | head -1 | tr ',' ' ' | awk '{print "kpis\t""'${model_list[i]}'""_loss_card8\t"$11}' | python _ce.py
else
   cat log_${config_list[${i}]}_8cards | grep "Epoch 0" | tail "-2" | head -1 | tr ',' ' ' | awk '{print "kpis\t""'${model_list[i]}'""_loss_card8\t"$10}' | python _ce.py
fi
cat log_${config_list[${i}]}_8cards | grep "Epoch 0" | tail "-1" | tr ',' ' ' | awk '{print "kpis\t""'${model_list[i]}'""_time_card8\t"$11}' | python _ce.py

done
