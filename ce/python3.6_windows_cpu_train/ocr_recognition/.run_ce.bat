@echo off
set ce_mode=1
rem train
python train.py --model=crnn_ctc --total_step=100 --save_model_period=100 --eval_period=100 --log_period=100 --save_model_dir=output_ctc --use_gpu=False | python _ce.py
rem train
python train.py --model=attention --total_step=20 --save_model_period=10 --eval_period=10 --save_model_dir=output_attention --use_gpu=False > %log_path%/ocr_attention_T.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\ocr_attention_T.log  %log_path%\FAIL\ocr_attention_T.log
        echo   ocr_attention,train,FAIL  >> %log_path%\result.log
        echo  training of ocr_attention failed!
) else (
        move  %log_path%\ocr_attention_T.log  %log_path%\SUCCESS\ocr_attention_T.log
        echo   ocr_attention,train,SUCCESS  >> %log_path%\result.log
        echo   training of ocr_attention successfully!
)
rem eval
python eval.py --model=crnn_ctc --model_path=output_ctc/model_00100 --use_gpu False > %log_path%/ocr_ctc_E.log
if not %errorlevel% == 0 (
        move  %log_path%\ocr_ctc_E.log  %log_path%\FAIL\ocr_ctc_E.log
        echo   ocr_ctc,eval,FAIL  >> %log_path%\result.log
        echo   evaling of ocr_CEC failed!
) else (
        move  %log_path%\ocr_ctc_E.log  %log_path%\SUCCESS\ocr_ctc_E.log
        echo   ocr_ctc,eval,SUCCESS  >> %log_path%\result.log
        echo   evaling of ocr_CEC successfully!
)
python eval.py --model=attention --model_path=output_attention/model_00010 --use_gpu False > %log_path%/ocr_attention_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\ocr_attention_E.log  %log_path%\FAIL\ocr_attention_E.log
        echo   ocr_attention,eval,FAIL  >> %log_path%\result.log
        echo  evaling of ocr_attention failed!
) else (
        move  %log_path%\ocr_attention_E.log  %log_path%\SUCCESS\ocr_attention_E.log
        echo   ocr_attention,eval,SUCCESS  >> %log_path%\result.log
        echo   evaling of ocr_attention successfully!
)
rem infer
mklink /j data %data_path%\ctc_data
python iner.py --model=crnn_ctc --model_path=output_ctc/model_00100 --use_gpu False --input_images_dir=data/test_images --input_images_list=data/test.list > %log_path%/ocr_ctc_I.log
if not %errorlevel% == 0 (
        move  %log_path%\ocr_ctc_I.log  %log_path%\FAIL\ocr_ctc_I.log
        echo   ocr_ctc,infer,FAIL  >> %log_path%\result.log
        echo   infering of ocr_CTC failed!
) else (
        move  %log_path%\ocr_ctc_I.log  %log_path%\SUCCESS\ocr_ctc_I.log
        echo   ocr_ctc,infer,SUCCESS  >> %log_path%\result.log
        echo   infering of ocr_CTC successfully!
)
python infer.py --model=attention --model_path=output_attention/model_00010 --use_gpu False --input_images_dir=data/test_images --input_images_list=data/test.list > %log_path%/ocr_attention_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\ocr_attention_I.log  %log_path%\FAIL\ocr_attention_I.log
        echo   ocr_attention,infer,FAIL  >> %log_path%\result.log
        echo  infering of ocr_attention failed!
) else (
        move  %log_path%\ocr_attention_I.log  %log_path%\SUCCESS\ocr_attention_I.log
        echo   ocr_attention,infer,SUCCESS  >> %log_path%\result.log
        echo   infering of ocr_attention successfully!
)
