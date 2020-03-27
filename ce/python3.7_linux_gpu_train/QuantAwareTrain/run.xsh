#!/bin/bash
export test_code_dir="${PWD}/../../test"
export qat_models_dir="${test_code_dir}/inference_ce/python_gpu/QuantAwareTrain"
export pretrain_model_dir="${dataset_path}/QuantAwareTrain"

#copy models files
cp -r ${qat_models_dir}/. ./

if [ -d 'ILSVRC2012' ];then rm -rf ILSVRC2012
fi
ln -s ${pretrain_model_dir}/pretrain pretrain
ln -s ${pretrain_model_dir}/ILSVRC2012 ILSVRC2012

./.run_ce.sh
