#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r ${models_dir}/dygraph/transformer/. ./
if [ -d "wmt16_ende_data_bpe" ];then rm -rf wmt16_ende_data_bpe
fi
ln -s ${dataset_path}/dygraph_transformer/wmt16_ende_data_bpe

./.run_ce.sh
