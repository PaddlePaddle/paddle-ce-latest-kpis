#!/bin/bash
###############################################################
##                  注意-- 注意--注意                        ##
## 请先在config.sh文件中填好配置                             ##
## 请将下面的 job_id 替换成自己的作业ID                      ##
##                                                           ##
###############################################################

source ./config.sh

# set job_id which need to be state
job_id="job-e6c5b2362d4e1671"

paddlecloud job --server ${server} \
        --port ${port} \
        --user-ak ${user_ak} \
        --user-sk ${user_sk} \
        state $job_id 
