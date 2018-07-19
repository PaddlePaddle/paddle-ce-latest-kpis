#!/bin/bash
###############################################################
##                  注意-- 注意--注意                        ##
## 请先在config.sh文件中填好配置                             ##
## 请将下面的 cluster_name 替换成所在组关联的k8s集群名称     ##
##                                                           ##
###############################################################

source ./config.sh

cluster_name="yq01-jpaas-paddle01-cpu"

paddlecloud cluster --server ${server} \
        --port ${port} \
        --user-ak ${user_ak} \
        --user-sk ${user_sk} \
        info --cluster-name ${cluster_name}
