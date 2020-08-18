@echo off

set FLAGS_fast_eager_deletion_mode=1
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fraction_of_gpu_memory_to_use=0.98

set CUDA_VISIBLE_DEVICES=0
python train.py --config=./tsm.yaml --use_gpu=True --epoch=1 --batch_size=1 > tsm.log
if %errorlevel% GTR 0 (
        move  %log_path%\dygraph_tsm_T.log  %log_path%\FAIL\dygraph_tsm_T.log
        echo   dygraph_tsm,train,FAIL  >> %log_path%\result.log
        echo  train of dygraph_tsm failed!
) else (
        move  %log_path%\dygraph_tsm_T.log  %log_path%\SUCCESS\dygraph_tsm_T.log
        echo   dygraph_tsm,train,SUCCESS  >> %log_path%\result.log
        echo   train of dygraph_tsm successfully!
)





