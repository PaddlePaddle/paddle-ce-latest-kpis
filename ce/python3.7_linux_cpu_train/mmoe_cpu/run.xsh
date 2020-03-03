#!/bin/bash
export models_dir=$PWD/../../models_repo

#copy models files
cp -r ${models_dir}/PaddleRec/multi-task/MMoE/* ./

# auto data
./.run_ce.sh
