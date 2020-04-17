#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleRec/text_matching_on_quora/_ce.py
rm -rf ${models_dir}/PaddleRec/text_matching_on_quora/.run_ce.sh
cp -r ${models_dir}/PaddleRec/text_matching_on_quora/. ./
./.run_ce.sh
