#!/bin/bash
###############################################################
##                  注意-- 注意--注意                        ##
##                     mpi 作业演示                          ##
## 请将下面的 user_ak/user_sk 替换成自己的 ak/sk             ##
## 请将下面的 cluster_name 替换成所在组关联的MPI集群名称     ##
##                                                           ##
###############################################################

cur_time=`date  +"%Y%m%d%H%M"` 
job_name=fit_a_line_demo_${cur_time}

# 测试环境地址
# server="yq01-rdqa-bml27.yq01.baidu.com"
# port=8089

# 线上正式环境
server="paddlecloud.baidu-int.com"
port=80

# 请替换成所在组下的个人 access key
user_ak="xxxxx"
# 请替换成所在组下的个人 secret key
user_sk="xxxxx"
# 请替换成所在组关联的 MPI 集群名称
cluster_name="xxxxx"
# 作业版本
job_version="paddle-v2-v0.10"
# 启动命令
start_cmd="python train.py"


paddlecloud job --server ${server} \
        --port ${port} \
        --user-ak ${user_ak} \
        --user-sk ${user_sk} \
        train --job-name ${job_name} \
        --job-conf config.ini \
        --start-cmd "${start_cmd}" \
        --files end_hook.sh before_hook.sh train.py \
        --cluster-name ${cluster_name} \
        --job-version ${job_version} \
        --mpi-nodes 1
