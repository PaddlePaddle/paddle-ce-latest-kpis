#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r ${models_dir}/PaddleCV/Paddle3D/PointNet++/. ./
if [ -d "dataset" ];then rm -rf dataset
fi
ln -s ${dataset_path}/Paddle3D/PointNet++/dataset dataset
cd ext_op/src/
bash make.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python -c 'import paddle; print(paddle.sysconfig.get_lib())'`
cd ..
export PYTHONPATH=$PYTHONPATH:`pwd`
python tests/test_farthest_point_sampling_op.py

if [ $? -ne 0 ];then
    echo "build pointnet2_lib.so failed!"
    exit 1
fi

cd ..

./.run_ce.sh
