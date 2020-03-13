@echo off
rem This file is only used for continuous evaluation.
set FLAGS_fast_eager_deletion_mode=1
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fraction_of_gpu_memory_to_use=0.98
set CUDA_VISIBLE_DEVICES=0
python train.py --use_multiprocess=False --snapshot_iter 100 --max_iter 200 --batch_size 1 --use_gpu true > dygraph_yolov3.log 2>&1
type dygraph_yolov3.log|grep "Iter"|awk -F "[ ,]" "END{print \"kpis\ttrain_loss\t\"$5}"|python _ce.py
type dygraph_yolov3.log|grep "Iter"|awk -F "[ ,]" "END{print \"kpis\ttrain_time\t\"$8}"|python _ce.py
