#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleNLP/PaddleLARK/ELMo/. ./

./.run_ce.sh


