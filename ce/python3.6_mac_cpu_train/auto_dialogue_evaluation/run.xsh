#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleNLP/PaddleDialogue/auto_dialogue_evaluation/. ./

rm -rf data
ln -s ${data_path}/auto_dialogue_evaluation/data data

./.run_ce.sh
