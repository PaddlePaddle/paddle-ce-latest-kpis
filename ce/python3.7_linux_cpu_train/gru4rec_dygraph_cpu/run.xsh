#!/bin/bash
export models_dir=$PWD/../../models_repo

#copy models files
cp -r ${models_dir}/PaddleRec/gru4rec/dy_graph/* ./

rm -rf data
ln -s ${dataset_path}/gru4rec_dygraph/data data
./.run_ce.sh
