#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleRec/text_matching_on_quora/_ce.py
rm -rf ${models_dir}/PaddleRec/text_matching_on_quora/.run_ce.sh
cp -r ${models_dir}/PaddleRec/text_matching_on_quora/. ./
cd /$HOME/.cache/paddle/dataset
if [ -d "Quora_question_pair_partition" ];then rm -rf Quora_question_pair_partition
fi
ln -s ${dataset_path}/text_matching_on_quora/Quora_question_pair_partition Quora_question_pair_partition
ln -s ${dataset_path}/text_matching_on_quora/glove.840B.300d.txt glove.840B.300d.txt
cd -
./.run_ce.sh
