#/bin/bash
#export models_dir=$PWD/../../models_repo

#49
export models_dir=/paddle/rec/ce_model/models
export dataset_path=/paddle/all_data/rec

#copy models files
cp -r ${models_dir}/PaddleRec/ctr/dcn/* ./
if [ -d "data" ];then rm -rf data
fi
ln -s ${dataset_path}/dcn data

./.run_ce.sh
