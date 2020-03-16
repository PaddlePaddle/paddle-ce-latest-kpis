#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/dygraph/ptb_lm/_ce.py
rm -rf ${models_dir}/dygraph/ptb_lm/.run_ce.sh
cp -r ${models_dir}/dygraph/ptb_lm/. ./
if [ -d "simple-examples" ];then rm -rf simple-examples
fi
ln -s ${dataset_path}/dygraph_ptb_lm/simple-examples simple-examples

./.run_ce.sh
