#!/bin/bash
###############################################################
##                  注意-- 注意--注意                        ##
## 请先在config.sh文件中填好配置                             ##
## 请将下面的 job_id 替换成作业ID                            ##
##                                                           ##
###############################################################

source ./config.sh

# set job_id first 
job_id="job-xxxx"

paddlecloud job --server ${server} \
        --port ${port} \
        --user-ak ${user_ak} \
        --user-sk ${user_sk} \
        info ${job_id}
