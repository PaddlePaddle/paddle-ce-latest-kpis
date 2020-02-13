#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleCV/icnet/. ./

./.run_ce.sh
