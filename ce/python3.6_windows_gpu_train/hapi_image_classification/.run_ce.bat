@echo off
rem This file is only used for continuous evaluation.
set FLAGS_cudnn_deterministic=True
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
set FLAGS_fraction_of_gpu_memory_to_use=0

set CUDA_VISIBLE_DEVICES=0
set PATH=C:\Program Files (x86)\GnuWin32\bin;%PATH%

setlocal enabledelayedexpansion
for %%I in (mobilenet_v1 mobilenet_v2 resnet50 vgg16) do (
rem for %%I in (mobilenet_v1) do (
python main.py --arch %%I -d --epoch 1 --output-dir=dygraph data/ILSVRC2012 -b 32 > hapi_%%I_d.log
python main.py --arch %%I --epoch 1 --output-dir=static data/ILSVRC2012 -b 32 > hapi_%%I_s.log
type hapi_%%I_d.log |grep "step 16/16"|gawk -F "[:-]" "NR==1{print \"kpis\t%%I_train_loss\t\"$3}" | python _ce.py 
type hapi_%%I_d.log |grep "step 16/16"|gawk -F "[:-]" "END{print \"kpis\t%%I_eval_loss\t\"$3}" | python _ce.py 
rem eval
python main.py --arch %%I -d --epoch 1 --output-dir=dygraph --eval_only -r dygraph/0 data/ILSVRC2012 >%log_path%/hapi_%%I_E.log 2>&1
if not !errorlevel! == 0 (
        move  %log_path%\hapi_%%I_E.log  %log_path%\FAIL\hapi_%%I_E.log
        echo   hapi_%%I,eval,FAIL  >> %log_path%\result.log
        echo  evaling of hapi_%%I failed!
) else (
        move  %log_path%\hapi_%%I_E.log  %log_path%\SUCCESS\hapi_%%I_E.log
        echo   hapi_%%I,eval,SUCCESS  >> %log_path%\result.log
        echo   evaling of hapi_%%I successfully!
)

)
