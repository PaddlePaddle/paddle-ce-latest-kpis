#!/bin/bash
export test_code_dir="${PWD}/../../test"
export qat_models_dir="${test_code_dir}/inference_ce/python_gpu/QuantAwareTrain"
export pretrain_model_dir="${dataset_path}/QuantAwareTrain"

#copy models files
cp -r ${qat_models_dir}/* ./

if [ -d 'dataset/ILSVRC2012' ];then rm -rf dataset
fi
ln -s ${pretrain_model_dir}/pretrain pretrain

mkdir dataset
ln -s ${pretrain_model_dir}/ILSVRC2012 dataset/ILSVRC2012

./.run_ce.sh
