#!/bin/bash
###############################################################
##                  注意-- 注意--注意                        ##
## 请先在config.sh文件中填好配置                             ##
## 请将下面的提交作业的配置补充完整                          ##
##                                                           ##
###############################################################

source ./config.sh

cur_time=`date  +"%Y%m%d%H%M"`
job_name=demo_job${cur_time}
cluster_name="yq01-jpaas-paddle01-cpu"
job_version="paddle-fluid-v0.13"
start_cmd="python train.py"
k8s_cpu_cores=2
k8s_wall_time="10:00:00"
k8s_memory="10Gi"
k8s_priority="high"

paddlecloud job --server ${server} \
        --port ${port} \
        --user-ak ${user_ak} \
        --user-sk ${user_sk} \
        train --job-name ${job_name} \
        --job-conf config.ini \
        --start-cmd "${start_cmd}" \
        --files before_hook.sh end_hook.sh train.py \
        --cluster-name ${cluster_name} \
        --job-version ${job_version}  \
        --k8s-cpu-cores ${k8s_cpu_cores} \
        --k8s-priority ${k8s_priority} \
        --k8s-wall-time ${k8s_wall_time} \
        --k8s-memory ${k8s_memory} \
        --json
