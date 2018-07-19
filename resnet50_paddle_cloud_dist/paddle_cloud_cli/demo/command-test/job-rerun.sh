#!/bin/bash
###############################################################
##                  注意-- 注意--注意                        ##
## 请先在config.sh文件中填好配置                             ##
## 请填写需要重跑的作业ID列表                                ##
##                                                           ##
###############################################################

source ./config.sh

# set job_ids which need to rerun
job_id1="job-e6c5b1e12df2f4c0"
job_id2="job-e6c5b1e167790222"

paddlecloud job --server ${server} \
        --port ${port} \
        --user-ak ${user_ak} \
        --user-sk ${user_sk} \
        rerun ${job_id1},${job_id2}
