#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleCV/ssd/.run_ce.sh
rm -rf ${models_dir}/PaddleCV/ssd/_ce.py
cp -r ${models_dir}/PaddleCV/ssd/. ./

./.run_ce.sh
