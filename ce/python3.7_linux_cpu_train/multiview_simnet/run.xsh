#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleRec/multiview_simnet/.run_ce.sh
cp -r ${models_dir}/PaddleRec/multiview_simnet/. ./

./.run_ce.sh
