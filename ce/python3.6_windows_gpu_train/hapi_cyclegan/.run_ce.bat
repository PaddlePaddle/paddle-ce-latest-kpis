@echo off
rem This file is only used for continuous evaluation.

rem train
python train.py --checkpoint_path=static --epoch=1 > hapi_cyclegan_static.log 
python train.py -d --checkpoint_path=dygraph --epoch=1 > hapi_cyclegan_dygraph.log 
type hapi_cyclegan_dygraph.log  |gawk -F "[|:] " "END{print \"kpis\ttrain_G_loss\t\"$6}" | python _ce.py
type hapi_cyclegan_dygraph.log  |gawk -F "[|:] " "END{print \"kpis\ttrain_DA_loss\t\"$8}" | python _ce.py
type hapi_cyclegan_dygraph.log  |gawk -F "[|:] " "END{print \"kpis\ttrain_DB_loss\t\"$10}" | python _ce.py

rem eval
python test.py --init_model=dygraph/0 >%log_path%/hapi_cyclegan_dygraph_E.log 2>&1
if %errorlevel% GTR 0 (
        move  %log_path%\hapi_cyclegan_dygraph_E.log  %log_path%\FAIL\hapi_cyclegan_dygraph_E.log
        echo   hapi_cyclegan_dygraph,eval,FAIL  >> %log_path%\result.log
        echo  evaling of hapi_cyclegan_dygraph failed!
) else (
        move  %log_path%\hapi_cyclegan_dygraph_E.log  %log_path%\SUCCESS\hapi_cyclegan_dygraph_E.log
        echo   hapi_cyclegan_dygraph,eval,SUCCESS  >> %log_path%\result.log
        echo   evaling of hapi_cyclegan_dygraph successfully!
)

rem infer
python infer.py --init_model=dygraph/0 --input=./image/testA/123_A.jpg --input_style=A >%log_path%/hapi_cyclegan_dygraph_I.log 2>&1
if %errorlevel% GTR 0 (
        move  %log_path%\hapi_cyclegan_dygraph_I.log  %log_path%\FAIL\hapi_cyclegan_dygraph_I.log
        echo   hapi_cyclegan_dygraph,infer,FAIL  >> %log_path%\result.log
        echo  infering of hapi_cyclegan_dygraph failed!
) else (
        move  %log_path%\hapi_cyclegan_dygraph_I.log  %log_path%\SUCCESS\hapi_cyclegan_dygraph_I.log
        echo   hapi_cyclegan_dygraph,infer,SUCCESS  >> %log_path%\result.log
        echo   infering of hapi_cyclegan_dygraph successfully!
)
