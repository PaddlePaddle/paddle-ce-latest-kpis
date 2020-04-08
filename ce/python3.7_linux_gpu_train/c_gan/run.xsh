#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleCV/gan/c_gan/.run_ce.sh
cp -r ${models_dir}/PaddleCV/gan/c_gan/. ./

./.run_ce.sh
