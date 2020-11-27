#!/bin/bash
export rec_dir=$PWD/../../rec_repo
#copy models files
cp -r ${rec_dir}/. ./
./.run_ce.sh
