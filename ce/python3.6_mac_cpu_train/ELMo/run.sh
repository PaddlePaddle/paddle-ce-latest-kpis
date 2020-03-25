#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleNLP/pretrain_language_models/ELMo/. ./

./.run_ce.sh


