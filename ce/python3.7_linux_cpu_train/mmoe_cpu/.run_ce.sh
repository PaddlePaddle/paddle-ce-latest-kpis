#!/bin/bash

# cpu

time (python mmoe_train.py >mmoe_cpu_T.log) >>mmoe_cpu_T.log 2>&1
if [ $? -ne 0 ];then
	echo -e "mmoe_cpu_T,train,FAIL"
else
	echo -e "mmoe_cpu_T,train,SUCCESS"
fi
