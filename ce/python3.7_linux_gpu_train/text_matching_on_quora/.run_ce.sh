#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
rm -rf *_factor.txt
# gpu 1card
# CDSSM
export CUDA_VISIBLE_DEVICES=0

python train_and_evaluate.py --model_name=cdssmNet --config=cdssm_base --enable_ce --epoch_num 1 >quora_cdssmNet_gpu1.log 2>&1
sed -i "s/each_pass_duration_card1/cdssmNet_each_pass_duration_card1/g" quora_cdssmNet_gpu1.log
sed -i "s/train_avg_cost_card1/cdssmNet_train_avg_cost_card1/g" quora_cdssmNet_gpu1.log
sed -i "s/train_avg_acc_card1/cdssmNet_train_avg_acc_card1/g" quora_cdssmNet_gpu1.log
cat quora_cdssmNet_gpu1.log | python _ce.py
# DecAttNet
python train_and_evaluate.py --model_name=DecAttNet --config=decatt_glove --enable_ce --epoch_num 1 >quora_DecAttNet_gpu1.log 2>&1
sed -i "s/each_pass_duration_card1/DecAttNet_each_pass_duration_card1/g" quora_DecAttNet_gpu1.log
sed -i "s/train_avg_cost_card1/DecAttNet_train_avg_cost_card1/g" quora_DecAttNet_gpu1.log
sed -i "s/train_avg_acc_card1/DecAttNet_train_avg_acc_card1/g" quora_DecAttNet_gpu1.log
cat quora_DecAttNet_gpu1.log | python _ce.py
# InferSentNet_v1
python train_and_evaluate.py --model_name=InferSentNet --config=infer_sent_v1 --enable_ce --epoch_num 1 >quora_InferSentNet_v1_gpu1.log 2>&1
sed -i "s/each_pass_duration_card1/InferSentNet_v1_each_pass_duration_card1/g" quora_InferSentNet_v1_gpu1.log
sed -i "s/train_avg_cost_card1/InferSentNet_v1_train_avg_cost_card1/g" quora_InferSentNet_v1_gpu1.log
sed -i "s/train_avg_acc_card1/InferSentNet_v1_train_avg_acc_card1/g" quora_InferSentNet_v1_gpu1.log
cat quora_InferSentNet_v1_gpu1.log | python _ce.py
# InferSentNet_v2
python train_and_evaluate.py --model_name=InferSentNet --config=infer_sent_v2 --enable_ce --epoch_num 1 >quora_InferSentNet_v2_gpu1.log 2>&1
sed -i "s/each_pass_duration_card1/InferSentNet_v2_each_pass_duration_card1/g" quora_InferSentNet_v2_gpu1.log
sed -i "s/train_avg_cost_card1/InferSentNet_v2_train_avg_cost_card1/g" quora_InferSentNet_v2_gpu1.log
sed -i "s/train_avg_acc_card1/InferSentNet_v2_train_avg_acc_card1/g" quora_InferSentNet_v2_gpu1.log
cat quora_InferSentNet_v2_gpu1.log | python _ce.py
# SSENet
python train_and_evaluate.py --model_name=SSENet --config=sse_base --enable_ce --epoch_num 1 >quora_SSENet_gpu1.log 2>&1
sed -i "s/each_pass_duration_card1/SSENet_each_pass_duration_card1/g" quora_SSENet_gpu1.log
sed -i "s/train_avg_cost_card1/SSENet_train_avg_cost_card1/g" quora_SSENet_gpu1.log
sed -i "s/train_avg_acc_card1/SSENet_train_avg_acc_card1/g" quora_SSENet_gpu1.log
cat quora_SSENet_gpu1.log | python _ce.py
