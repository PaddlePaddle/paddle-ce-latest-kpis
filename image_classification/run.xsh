#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
cudaid=${multi_se_resnext_cudaid:=0,1,2,3} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true python train.py --batch_size=64
mv train_speed_kpi_factor.txt four_card_train_speed_kpi_factor.txt

cudaid=${se_resnext_cudaid:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true python train.py --batch_size=64
