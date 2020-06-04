@echo off 
set FLAGS_sync_nccl_allreduce=0
set FLAGS_eager_delete_tensor_gb=1.0

set CUDA_VISIBLE_DEVICES=0
rem train
python -u run_classifier.py --use_cuda true --cpu_num 3 --do_train true --do_eval false --do_test false --build_dict false --data_dir ./data/input/ --save_dir ./data/output/ --config_path  ./data/input/model.conf --batch_size 64 --init_checkpoint None > %log_path%/domain_classification_T.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\domain_classification_T.log  %log_path%\FAIL\domain_classification_T.log
        echo   domain_classification,train,FAIL  >> %log_path%\result.log
        echo   train of domain_classification failed!
) else (
        move  %log_path%\domain_classification_T.log  %log_path%\SUCCESS\domain_classification_T.log
        echo   domain_classification,train,SUCCESS  >> %log_path%\result.log
        echo   train of domain_classification successfully!
 )
rem eval
python -u run_classifier.py --use_cuda true --cpu_num 3 --do_train true --do_eval true --do_test false --build_dict false --data_dir ./data/input/ --save_dir ./data/output/ --config_path  ./data/input/model.conf --batch_size 64 --init_checkpoint None > %log_path%/domain_classification_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\domain_classification_E.log  %log_path%\FAIL\domain_classification_E.log
        echo   domain_classification,eval,FAIL  >> %log_path%\result.log
        echo   eval of domain_classification failed!
) else (
        move  %log_path%\domain_classification_E.log  %log_path%\SUCCESS\domain_classification_E.log
        echo   domain_classification,eval,SUCCESS  >> %log_path%\result.log
        echo   eval of domain_classification successfully!
 )
rem infer
python -u run_classifier.py --use_cuda true --cpu_num 3 --do_train false --do_eval false --do_test true --build_dict false --data_dir ./data/input/ --save_dir ./data/output/ --config_path  ./data/input/model.conf --batch_size 64 --init_checkpoint None > %log_path%/domain_classification_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\domain_classification_I.log  %log_path%\FAIL\domain_classification_I.log
        echo   domain_classification,infer,FAIL  >> %log_path%\result.log
        echo   infer of domain_classification failed!
) else (
        move  %log_path%\domain_classification_I.log  %log_path%\SUCCESS\domain_classification_I.log
        echo   domain_classification,infer,SUCCESS  >> %log_path%\result.log
        echo   infer of domain_classification successfully!
 )
