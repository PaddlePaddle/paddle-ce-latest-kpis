#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleCV/PaddleGAN/cycle_gan/.run_ce.sh
cp -r ${models_dir}/PaddleCV/PaddleGAN/cycle_gan/. ./

./.run_ce.sh
