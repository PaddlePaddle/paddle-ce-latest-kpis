#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleRec/gnn/. ./
if [ -d "data" ]; then
rm -rf data
fi
ln -s $data_path/gnn data
./.run_ce.sh
