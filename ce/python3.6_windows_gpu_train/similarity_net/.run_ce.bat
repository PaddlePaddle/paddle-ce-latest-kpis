@echo off
set FLAGS_enable_parallel_graph=1
set FLAGS_sync_nccl_allreduce=1
set FLAGS_fraction_of_gpu_memory_to_use=0.95
set CUDA_VISIBLE_DEVICES=0 
rem train
python run_classifier.py --task_name simnet --use_cuda true --do_train True --do_valid True --do_test False --do_infer False --batch_size 128 --train_data_dir ./data/zhidao --valid_data_dir ./data/zhidao --test_data_dir ./data/zhidao --infer_data_dir ./data/zhidao --output_dir ./model_files --config_path  ./config/bow_pairwise.json --vocab_path ./data/term2id.dict --epoch 6 --save_steps 1000 --validation_steps 1 --compute_accuracy True --lamda 0.958 --task_mode pairwise --init_checkpoint ""  --enable_ce | python _ce.py				
rem eval
python run_classifier.py --task_name simnet --use_cuda true --do_test True --verbose_result True --batch_size 128 --test_data_dir ./data/test_pairwise_data --test_result_path ./test_result --config_path ./config/bow_pairwise.json --vocab_path ./data/term2id.dict --task_mode pairwise --compute_accuracy False --lamda 0.95 --init_checkpoint ./model_files/simnet_bow_pairwise_pretrained_model/ > %log_path%/simnet_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\simnet_E.log  %log_path%\FAIL\simnet_E.log
        echo   similarity_net,eval,FAIL  >> %log_path%\result.log
        echo   eval of similarity_net failed!
) else (
        move  %log_path%\simnet_E.log  %log_path%\SUCCESS\simnet_E.log
        echo   similarity_net,eval,SUCCESS  >> %log_path%\result.log
        echo   eval of similarity_net successfully!
)
rem infer
python run_classifier.py --task_name simnet --use_cuda true --do_infer True --batch_size 128 --infer_data_dir ./data/infer_data --infer_result_path ./infer_result --config_path ./config/bow_pairwise.json --vocab_path ./data/term2id.dict --task_mode pairwise --init_checkpoint ./model_files/simnet_bow_pairwise_pretrained_model/ > %log_path%/simnet_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\simnet_I.log  %log_path%\FAIL\simnet_I.log
        echo   similarity_net,infer,FAIL  >> %log_path%\result.log
        echo  infer of similarity_net failed!
) else (
        move  %log_path%\simnet_I.log  %log_path%\SUCCESS\simnet_I.log
        echo   similarity_net,infer,SUCCESS  >> %log_path%\result.log
        echo   infer of similarity_net successfully!
)