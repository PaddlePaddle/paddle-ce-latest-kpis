#!/bin/bash
###############################################################
##                  注意-- 注意--注意                        ##
## 请先在config.sh文件中填好配置                             ##
##                                                           ##
###############################################################

source ./config.sh

paddlecloud job --server ${server} \
        --port ${port} \
        --user-ak ${user_ak} \
        --user-sk ${user_sk} \
        list --type normal --page 1 --size 15 \
        #--json
