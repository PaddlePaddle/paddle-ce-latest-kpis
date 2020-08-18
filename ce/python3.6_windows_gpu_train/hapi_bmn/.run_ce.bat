@echo off

set FLAGS_fast_eager_deletion_mode=1
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fraction_of_gpu_memory_to_use=0.98

set CUDA_VISIBLE_DEVICES=0
rem train
python train.py --config=./bmn.yaml --device gpu --epoch=1 > hapi_bmn_static.log
python train.py --config=./bmn.yaml --device gpu --epoch=1 -d > hapi_bmn_dygraph.log
type hapi_bmn_dygraph.log|grep "step 1182/1182"|gawk -F "[:-]" "{print \"kpis\ttrain_loss\t\"$3}"| python _ce.py
rem eval
python eval.py --weights=checkpoint/0 > %log_path%/hapi_bmn_E.log
if %errorlevel% GTR 0 (
        move  %log_path%\hapi_bmn_E.log  %log_path%\FAIL\hapi_bmn_E.log
        echo   hapi_bmn,eval,FAIL  >> %log_path%\result.log
        echo  evaling of hapi_bmn failed!
) else (
        move  %log_path%\hapi_bmn_E.log  %log_path%\SUCCESS\hapi_bmn_E.log
        echo   hapi_bmn,eval,SUCCESS  >> %log_path%\result.log
        echo   evaling of hapi_bmn successfully!
)
rem infer
python predict.py --weights=checkpoint/0 --filelist=infer.list > %log_path%/hapi_bmn_I.log
if %errorlevel% GTR 0 (
        move  %log_path%\hapi_bmn_I.log  %log_path%\FAIL\hapi_bmn_I.log
        echo   hapi_bmn,infer,FAIL  >> %log_path%\result.log
        echo  infering of hapi_bmn failed!
) else (
        move  %log_path%\hapi_bmn_I.log  %log_path%\SUCCESS\hapi_bmn_I.log
        echo   hapi_bmn,infer,SUCCESS  >> %log_path%\result.log
        echo   infering of hapi_bmn successfully!
)




