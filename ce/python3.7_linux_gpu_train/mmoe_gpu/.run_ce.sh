#!/bin/bash

# gpu1
export CUDA_VISIBLE_DEVICES=3

time (python mmoe_train.py >mmoe_gpu1_T.log) >>mmoe_gpu1_T.log 2>&1
if [ $? -ne 0 ];then
	echo -e "mmoe_gpu1_T,train,FAIL"
else
	echo -e "mmoe_gpu1_T,train,SUCCESS"
fi
# gpu8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

time (python mmoe_train.py >mmoe_gpu8_T.log) >>mmoe_gpu8_T.log 2>&1
if [ $? -ne 0 ];then
	echo -e "mmoe_gpu8_T,train,FAIL"
else
	echo -e "mmoe_gpu8_T,train,SUCCESS"
fi