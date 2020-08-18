@echo off
rem This file is only used for continuous evaluation.

rem train
python train.py --checkpoint_path=static --epoch=1 > hapi_ocr_static.log 
python train.py --dynamic True --checkpoint_path=dygraph --epoch=1 > hapi_ocr_dygraph.log
type hapi_ocr_static.log |grep "step 12482/12482"|gawk -F "[-:] " "END{print \"kpis\ttrain_loss\t\"$3}" | python _ce.py
type hapi_ocr_static.log |grep "step 63/63"|gawk -F "[-:] " "END{print \"kpis\teval_loss\t\"$3}"| python _ce.py
rem eval
python eval.py --init_model=dygraph/0 --dynamic True  >%log_path%/hapi_ocr_dygraph_E.log 2>&1
if %errorlevel% GTR 0 (
        move  %log_path%\hapi_ocr_dygraph_E.log  %log_path%\FAIL\hapi_ocr_dygraph_E.log
        echo   hapi_ocr_dygraph,eval,FAIL  >> %log_path%\result.log
        echo  evaling of hapi_ocr_dygraph failed!
) else (
        move  %log_path%\hapi_ocr_dygraph_E.log  %log_path%\SUCCESS\hapi_ocr_dygraph_E.log
        echo   hapi_ocr_dygraph,eval,SUCCESS  >> %log_path%\result.log
        echo   evaling of hapi_ocr_dygraph successfully!
)

rem infer
python predict.py --init_model=dygraph/0 --image_path=./images --dynamic=False --beam_size=3 >%log_path%/hapi_ocr_I.log 2>&1
if %errorlevel% GTR 0 (
        move  %log_path%\hapi_ocr_I.log  %log_path%\FAIL\hapi_ocr_I.log
        echo   hapi_ocr,infer,FAIL  >> %log_path%\result.log
        echo  infering of hapi_ocr failed!
) else (
        move  %log_path%\hapi_ocr_I.log  %log_path%\SUCCESS\hapi_ocr_I.log
        echo   hapi_ocr,infer,SUCCESS  >> %log_path%\result.log
        echo   infering of hapi_ocr successfully!
)
