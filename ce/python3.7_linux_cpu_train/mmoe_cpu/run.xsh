#!/bin/bash
# export models_dir=$PWD/../../models_repo

# 49
export models_dir=/paddle/rec/ce_test/models

#copy models files
cp -r ${models_dir}/PaddleRec/multi-task/MMoE/* ./

# auto data
./.run_ce.sh
