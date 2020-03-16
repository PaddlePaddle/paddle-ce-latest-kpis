@echo off
set FLAGS_enable_parallel_graph=1
set FLAGS_sync_nccl_allreduce=1
set FLAGS_fraction_of_gpu_memory_to_use=0.95
set CUDA_VISIBLE_DEVICES=0 
rem train
python run_classifier.py --task_name simnet --use_cuda false --do_train True --do_valid True --do_test True --do_infer False --batch_size 128 --train_data_dir ./lcqmc_data/train.tsv --valid_data_dir ./lcqmc_data/dev.tsv --test_data_dir ./lcqmc_data/test.tsv --infer_data_dir ./lcqmc_data/test.tsv --output_dir ./model_files --config_path  ./config/bow_pointwise.json --vocab_path ./lcqmc_data/term2id.dict --epoch 100 --save_steps 200000 --validation_steps 200 --compute_accuracy False --lamda 0.958 --task_mode pointwise --init_checkpoint ""  > %log_path%/similarity_net_T.log 2>&1
type %log_path%\similarity_net_T.log |grep "AUC of test is"|awk "END{print \"kpis\ttest_auc\t\"$NF}" | python _ce.py			
if not %errorlevel% == 0 (
        move  %log_path%\similarity_net_T.log  %log_path%\FAIL\similarity_net_T.log
        echo   similarity_net,train,FAIL  >> %log_path%\result.log
        echo   train of similarity_net failed!
) else (
        move  %log_path%\similarity_net_T.log  %log_path%\SUCCESS\similarity_net_T.log
        echo   similarity_net,train,SUCCESS  >> %log_path%\result.log
        echo   train of similarity_net successfully!
)
