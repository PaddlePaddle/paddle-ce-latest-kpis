@echo off

set CUDA_VISIBLE_DEVICES=0
rem train
python train.py --num_epoch 1 > dygraph_deepfm.log 2>&1
type dygraph_deepfm.log | grep "finished and takes" | gawk "{print \"kpis\ttrain_duration\t\"$9}" | python _ce.py
type dygraph_deepfm.log | grep "test auc of epoch" | gawk "{print \"kpis\ttest_auc\t\"$9}"  | python _ce.py

